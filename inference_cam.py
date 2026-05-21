"""
Inférence en temps réel : reconnaissance de chiffres manuscrits via webcam.

Utilise directement le modèle LeNet entraîné (NumPy pur).
Aucune conversion TFLite, aucun pickle : on charge le .npz et on infère.

Usage:
    python inference_cam.py [chemin_modele.npz]

Contrôles:
    q : quitter
    + / - : ajuster la taille de la zone de capture (ROI)

Présentation attendue : chiffre au stylo noir sur papier blanc,
placé dans le carré vert.
"""

import sys
import collections
import numpy as np
import cv2

from model import LeNet
from preprocessing import preprocess_roi


# ============================================================================
# Lissage temporel des prédictions
# ============================================================================

class PredictionSmoother:
    """
    Moyenne les probabilités sur les N dernières frames.

    Sans lissage, la prédiction "clignote" d'une frame à l'autre à cause
    du bruit vidéo. En moyennant, on obtient un affichage stable et on
    réduit les faux positifs ponctuels.
    """

    def __init__(self, window=5):
        self.window = window
        self.history = collections.deque(maxlen=window)

    def update(self, probs):
        """probs: vecteur (10,). Retourne (classe_lissée, confiance_lissée)."""
        self.history.append(probs)
        avg = np.mean(self.history, axis=0)
        return int(np.argmax(avg)), float(np.max(avg))

    def reset(self):
        self.history.clear()


# ============================================================================
# Programme principal
# ============================================================================

def main(model_path='lenet_mnist.npz'):
    # --- Chargement du modèle ---
    print(f"Chargement du modèle depuis {model_path}...")
    model = LeNet()
    try:
        model.load(model_path)
    except FileNotFoundError:
        print(f"ERREUR: fichier '{model_path}' introuvable.")
        print("Lancez d'abord l'entraînement avec : python train.py")
        return

    # --- Ouverture de la webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR: impossible d'ouvrir la webcam.")
        return

    smoother = PredictionSmoother(window=5)
    roi_size = 200            # taille initiale de la zone de capture
    CONFIDENCE_THRESHOLD = 0.5  # en dessous, on n'affiche pas de prédiction ferme

    # Fenêtres déclarées à l'avance (évite des bugs d'affichage)
    cv2.namedWindow("Reconnaissance de chiffres", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Vue du modele (28x28)", cv2.WINDOW_NORMAL)

    print("Webcam active. Placez un chiffre dans le carré vert. 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture caméra.")
            break

        # Effet miroir : déplacements naturels à l'écran
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        # --- Définition de la ROI centrale ---
        x = (w_frame - roi_size) // 2
        y = (h_frame - roi_size) // 2
        # Copie de la zone AVANT de dessiner le rectangle (sinon le vert pollue)
        roi = frame[y:y+roi_size, x:x+roi_size].copy()

        # --- Prétraitement + inférence ---
        input_tensor, debug_img, found = preprocess_roi(roi)

        label_text = "Aucun chiffre"
        box_color = (0, 165, 255)  # orange = rien de net

        if found:
            probs = model.predict_proba(input_tensor)[0]  # (10,)
            pred, confidence = smoother.update(probs)

            if confidence >= CONFIDENCE_THRESHOLD:
                label_text = f"{pred}  ({confidence*100:.0f}%)"
                box_color = (0, 255, 0)  # vert = prédiction fiable
            else:
                label_text = f"{pred}?  ({confidence*100:.0f}%)"
                box_color = (0, 255, 255)  # jaune = incertain
        else:
            smoother.reset()

        # --- Dessin de l'interface ---
        cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), box_color, 2)
        cv2.putText(frame, label_text, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)
        cv2.putText(frame, "q: quitter   +/-: taille ROI", (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Reconnaissance de chiffres", frame)

        # Vue debug : ce que le modèle reçoit réellement, agrandi
        debug_large = cv2.resize(debug_img, (280, 280),
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Vue du modele (28x28)", debug_large)

        # --- Gestion clavier ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            roi_size = min(roi_size + 20, min(h_frame, w_frame))
            smoother.reset()
        elif key == ord('-') or key == ord('_'):
            roi_size = max(roi_size - 20, 60)
            smoother.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'lenet_mnist.npz'
    main(model_path)