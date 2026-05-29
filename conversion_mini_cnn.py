import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. Chargement du modèle NumPy...")
with open('mon_mini_cnn_numpy.pickle', 'rb') as f:
    params = pickle.load(f)

# Récupération des poids
conv_filters = params['conv_filters']    # Format NumPy: (8, 3, 3)
dense_weights = params['dense_weights']  # Format NumPy: (1352, 10)
dense_biases = params['dense_biases']    # Format NumPy: (10,)

print("2. Adaptation des dimensions pour Keras...")
# Keras attend les filtres sous la forme : (hauteur, largeur, canaux_entree, canaux_sortie)
# Notre NumPy a le format : (canaux_sortie, hauteur, largeur)
# 1. On transpose les axes (1, 2, 0) -> le (8, 3, 3) devient (3, 3, 8)
keras_conv_filters = np.transpose(conv_filters, (1, 2, 0))
# 2. On ajoute la dimension des canaux d'entrée (1 canal pour le gris) -> (3, 3, 1, 8)
keras_conv_filters = np.expand_dims(keras_conv_filters, axis=2)

print("3. Construction du moule Keras équivalent...")
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    # use_bias=False car notre implémentation NumPy n'avait pas de biais sur la convolution
    layers.Conv2D(8, (3, 3), use_bias=False, name='conv_custom'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    # use_bias=True car notre DenseSoftmax NumPy possédait un vecteur "self.biases"
    layers.Dense(10, activation='softmax', use_bias=True, name='dense_custom')
])

print("4. Injection des poids NumPy dans Keras...")
model.get_layer('conv_custom').set_weights([keras_conv_filters])
model.get_layer('dense_custom').set_weights([dense_weights, dense_biases])

print("5. Conversion en TFLite (Quantification INT8)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Données de calibration nécessaires pour passer en nombres entiers (INT8)
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# ATTENTION : On doit appliquer EXACTEMENT le même prétraitement que lors 
# de l'entraînement de notre modèle NumPy [-0.5, 0.5]
x_test = (x_test.astype(np.float32) / 255.0) - 0.5 
x_test = np.expand_dims(x_test, axis=-1)

def representative_data_gen():
    for i in range(100):
        yield [x_test[i:i+1]]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

out_path = 'mnist_classifier.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)

print(f"\nSuccès ! Modèle TFLite exporté sous '{out_path}'.")