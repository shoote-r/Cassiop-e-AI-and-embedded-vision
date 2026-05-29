import tensorflow as tf
import numpy as np

def build_and_convert_mnist_model():
    print("1. Chargement et prétraitement des données MNIST...")
    # Chargement du dataset depuis Keras
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalisation des pixels (0 à 1) et forçage du type en float32 pour TFLite
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Ajout d'une dimension pour le canal (les CNN attendent un format [batch, hauteur, largeur, canaux])
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("2. Construction de l'architecture du modèle CNN...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    print("3. Compilation du modèle...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("4. Entraînement du modèle (5 époques)...")
    # L'entraînement est rapide sur MNIST, même sans GPU
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    print("\n5. Évaluation des performances...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Précision sur les données de test : {accuracy * 100:.2f}%\n")

    print("6. Conversion du modèle en TensorFlow Lite (.tflite)...")
    # Initialisation du convertisseur à partir du modèle Keras en mémoire
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Application de l'optimisation par défaut (quantification) pour réduire la taille du fichier final
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Génération du modèle TFLite
    tflite_model = converter.convert()

    print("7. Sauvegarde du fichier sur le disque...")
    tflite_filename = "mnist_classifier.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)

    print(f"\n✅ Succès total ! Le modèle optimisé a été sauvegardé sous : {tflite_filename}")

if __name__ == "__main__":
    # Exécute la fonction principale
    build_and_convert_mnist_model()