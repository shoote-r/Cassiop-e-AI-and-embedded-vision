import cv2
import numpy as np
import tensorflow as tf

# 1. Charger le modèle TFLite
MODEL_PATH = "mnist_classifier.tflite" # Remplacer par le vrai nom de ton fichier
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Récupérer les détails des tenseurs d'entrée et de sortie
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Vérifier la forme d'entrée attendue (généralement)
input_shape = input_details[0]['shape']

# 2. Initialiser la webcam
cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter le programme.")

# Déclaration des fenêtres AVANT la boucle pour éviter les bugs d'affichage
cv2.namedWindow("Webcam - Reconnaissance MNIST", cv2.WINDOW_NORMAL)
cv2.namedWindow("Vue du Modele (Pre-processing)", cv2.WINDOW_NORMAL)

while True:
    # Lire l'image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture de la caméra.")
        break

    # Effet miroir pour que l'affichage soit plus naturel
    frame = cv2.flip(frame, 1)

    # 3. Définir une zone d'intérêt (Region of Interest - ROI) au centre
    h_frame, w_frame, _ = frame.shape
    roi_size = 200
    x = int(w_frame/2 - roi_size/2)
    y = int(h_frame/2 - roi_size/2)
    
    # Extraire la zone pure AVANT de dessiner le rectangle vert
    roi = frame[y:y + roi_size, x:x + roi_size].copy()
    
    # Dessiner le rectangle dans lequel l'utilisateur doit placer son chiffre
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)

    # 4. Prétraitement de l'image (Crucial pour MNIST)
    # Passer en niveaux de gris
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # CORRECTION 1 : Décommenter le flou et peut-être l'augmenter un peu (ex: 7x7) 
    # pour bien lisser le grain de la webcam.
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # CORRECTION 2 : Utiliser 'blur' (et non 'gray') 
    # J'ai aussi passé le dernier chiffre (la constante C) de 5 à 15. 
    # Plus ce chiffre est grand, plus le fond sera forcé à devenir noir (éliminant le bruit).
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    
    # Dilatation pour épaissir le trait du stylo
    kernel = np.ones((5, 5), np.uint8) 
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # Algorithme de recadrage (Bounding Box) et de centrage
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Prendre le contour le plus grand (pour ignorer le petit bruit)
        c = max(contours, key=cv2.contourArea)
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        
        # Si le contour est assez grand (pas juste un point de poussière)
        if w_c > 15 and h_c > 15:
            # Rogner l'image autour du chiffre
            digit = thresh[y_c:y_c+h_c, x_c:x_c+w_c]
            
            # Redimensionner le chiffre pour qu'il rentre dans une boîte de 20x20
            if w_c > h_c:
                new_w = 20
                new_h = int((h_c / w_c) * 20)
            else:
                new_h = 20
                new_w = int((w_c / h_c) * 20)
                
            # Éviter les erreurs de dimension à 0
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Créer le fond noir 28x28 et placer le chiffre au centre (Padding)
            pad_top = (28 - new_h) // 2
            pad_bottom = 28 - new_h - pad_top
            pad_left = (28 - new_w) // 2
            pad_right = 28 - new_w - pad_left
            
            resized = cv2.copyMakeBorder(digit_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        else:
            resized = cv2.resize(thresh, (28, 28))
    else:
        resized = cv2.resize(thresh, (28, 28))
    
    # Adoucir les bords (Anti-aliasing) pour coller au style MNIST
    resized = cv2.GaussianBlur(resized, (3, 3), 0)

    # Normaliser les valeurs entre 0 et 1 et forcer le type float32
    normalized = resized.astype('float32') / 255.0 - 0.5

    # Remodeler l'image pour qu'elle corresponde à ce que le modèle attend
    input_data = np.reshape(normalized, input_shape)

    # 5. Faire l'inférence
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Récupérer l'index ayant la plus haute probabilité
    prediction = np.argmax(output_data)
    confidence = np.max(output_data) * 100

    # 6. Afficher les résultats
    texte = f"Prediction: {prediction} ({confidence:.1f}%)"
    cv2.putText(frame, texte, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam - Reconnaissance MNIST", frame)
    
    # MODIFICATION 2 : Agrandir l'image finale 28x28 pour voir exactement ce que le modèle analyse
    debug_view = cv2.resize(resized, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Vue du Modele (Pre-processing)", debug_view)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()