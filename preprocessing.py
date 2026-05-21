"""
Prétraitement d'une image webcam vers le format attendu par le modèle MNIST.

Objectif : transformer une photo de chiffre (stylo noir sur papier blanc)
en une image 28x28 ressemblant le plus possible à un échantillon MNIST :
    - chiffre BLANC sur fond NOIR
    - chiffre recadré, centré
    - normalisé dans [0, 1]

Ce module isole toute la logique qui dépend d'OpenCV dans des fonctions
clairement délimitées, pour pouvoir tester le reste sans webcam.

Pourquoi reproduire le "style MNIST" ?
    MNIST a été construit ainsi : chiffres normalisés dans une boîte 20x20,
    puis centrés par centre de masse dans une image 28x28. Plus notre
    prétraitement s'en approche, plus le modèle (entraîné sur MNIST) sera fiable.
"""

import numpy as np
import cv2


def to_grayscale_blurred(roi_bgr):
    """ROI couleur (BGR) -> niveaux de gris débruités."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def binarize(gray):
    """
    Seuillage adaptatif INVERSÉ.

    - Adaptatif : robuste aux variations d'éclairage sur la feuille
      (une ombre d'un côté ne fait pas basculer toute l'image).
    - Inversé (THRESH_BINARY_INV) : sur l'image d'origine le chiffre est
      SOMBRE sur fond CLAIR ; après inversion il devient CLAIR sur fond
      SOMBRE, comme MNIST.

    Retourne une image binaire uint8 (0 ou 255).
    """
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,   # taille du voisinage pour le seuil local (impair)
        C=5             # constante soustraite à la moyenne locale
    )
    return thresh


def clean_binary(thresh):
    """
    Nettoyage morphologique :
    - ouverture (érosion puis dilatation) pour supprimer le petit bruit
    - légère dilatation pour épaissir le trait (les chiffres MNIST sont gras)
    """
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    return dilated


def extract_digit_bbox(binary):
    """
    Trouve la bounding box du plus gros contour (= le chiffre).

    Retourne (x, y, w, h) ou None si aucun contour exploitable.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Rejet du bruit résiduel. Attention : un '1' est légitimement très
    # ÉTROIT (faible largeur) mais GRAND en hauteur. On ne peut donc pas
    # exiger w >= 15 ET h >= 15. On rejette un contour seulement s'il est
    # petit dans les DEUX dimensions (vrai point de poussière), ou si sa
    # plus grande dimension reste sous un minimum absolu.
    largest_side = max(w, h)
    if largest_side < 20:
        return None
    return (x, y, w, h)


def center_in_28x28(digit_crop):
    """
    Reproduit la normalisation MNIST :
    1. Redimensionne le chiffre pour qu'il tienne dans une boîte 20x20
       en CONSERVANT le ratio d'aspect (un '1' reste fin, un '0' reste large).
    2. Le place au centre d'une image 28x28 noire.
    3. Recentre par centre de masse (comme le vrai pipeline MNIST).

    Args:
        digit_crop: image binaire (uint8) recadrée sur le chiffre
    Returns:
        image 28x28 uint8
    """
    h, w = digit_crop.shape

    # --- Étape 1 : redimensionner dans une boîte 20x20, ratio conservé ---
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))

    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- Étape 2 : coller au centre d'un canevas 28x28 ---
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # --- Étape 3 : recentrage par centre de masse ---
    canvas = shift_by_center_of_mass(canvas)
    return canvas


def shift_by_center_of_mass(img28):
    """
    Décale l'image pour que le centre de masse des pixels allumés
    coïncide avec le centre géométrique (14, 14). C'est exactement
    ce que fait le pipeline MNIST original.
    """
    total = img28.sum()
    if total == 0:
        return img28

    ys, xs = np.nonzero(img28)
    weights = img28[ys, xs].astype(np.float64)
    cy = (ys * weights).sum() / weights.sum()
    cx = (xs * weights).sum() / weights.sum()

    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img28, M, (28, 28), borderValue=0)
    return shifted


def preprocess_roi(roi_bgr):
    """
    Pipeline complet : ROI couleur -> image 28x28x1 float32 normalisée,
    prête pour model.predict().

    Retourne :
        input_tensor: (1, 28, 28, 1) float32 dans [0,1]
        debug_img: l'image 28x28 uint8 (pour affichage)
        found: bool indiquant si un chiffre a été détecté
    """
    gray = to_grayscale_blurred(roi_bgr)
    binary = binarize(gray)
    binary = clean_binary(binary)

    bbox = extract_digit_bbox(binary)
    if bbox is None:
        # Pas de chiffre net : on renvoie un 28x28 noir
        img28 = np.zeros((28, 28), dtype=np.uint8)
        found = False
    else:
        x, y, w, h = bbox
        digit_crop = binary[y:y+h, x:x+w]
        img28 = center_in_28x28(digit_crop)
        found = True

    # Petit flou pour adoucir les bords (MNIST a des bords antialiasés)
    img28_soft = cv2.GaussianBlur(img28, (3, 3), 0)

    # Normalisation -> tenseur modèle
    normalized = img28_soft.astype(np.float32) / 255.0
    input_tensor = normalized.reshape(1, 28, 28, 1)

    return input_tensor, img28_soft, found