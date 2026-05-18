import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_weights_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Normaliser en liste de arrays
    if isinstance(data, dict):
        # Try common keys w0,w1,w2 or ordered numpy values
        if all(k in data for k in ['w0', 'w1', 'w2']):
            return [np.array(data['w0']), np.array(data['w1']), np.array(data['w2'])]
        # fallback: collect numpy arrays from dict values in insertion order
        vals = [np.array(v) for v in data.values() if isinstance(v, (np.ndarray, list, tuple))]
        if vals:
            return vals
        raise ValueError("Impossible d'identifier des poids valides dans le pickle (dict).")
    elif isinstance(data, (list, tuple)):
        return [np.array(v) for v in data]
    elif isinstance(data, np.ndarray):
        return [data]
    else:
        raise ValueError('Format de pickle non supporté pour les poids: %s' % type(data))


def build_model():
    return keras.Sequential([
        # 1. On fixe le batch à 1 et on indique explicitement une image 28x28 (1 canal)
        layers.Input(batch_shape=(1, 28, 28, 1)),
        # 2. On aplatit l'image en 784 pour qu'elle s'adapte à vos poids
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid', use_bias=False, name='dense_1'),
        layers.Dense(64, activation='sigmoid', use_bias=False, name='dense_2'),
        layers.Dense(10, activation='softmax', use_bias=False, name='dense_out')
    ])


def inject_weights(model, weights_list):
    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
    if len(weights_list) < len(dense_layers):
        raise ValueError('Nombre de jeux de poids (%d) < nombre de couches Dense (%d)'
                         % (len(weights_list), len(dense_layers)))

    for i, layer in enumerate(dense_layers):
        w = np.array(weights_list[i])
        # Keras Dense kernel shape: (in_dim, out_dim)
        kernel_shape = tuple(layer.kernel.shape)

        if w.shape == kernel_shape:
            kernel = w
        elif w.T.shape == kernel_shape:
            kernel = w.T
        else:
            try:
                kernel = w.reshape(kernel_shape)
                print(f'Reshape des poids {i} -> {kernel_shape}')
            except Exception:
                raise ValueError(f"Incompatible weight shape for layer {layer.name}: got {w.shape}, expected {kernel_shape}")

        layer.set_weights([kernel])


def convert_to_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print('Récupération de données MNIST pour la calibration...')
    (_, _), (x_test, _) = keras.datasets.mnist.load_data()
    
    # MODIFICATION ICI : On reshape en format image (10000, 28, 28, 1)
    x_test = (x_test / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)
    
    def representative_data_gen():
        for i in range(100):
            yield [x_test[i:i+1]]
            
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

def main(pickle_path='model_150_epoch.pickle', out_path='mnist_custom.tflite'):
    print('Chargement des poids depuis', pickle_path)
    weights = load_weights_from_pickle(pickle_path)

    print('Construction du modèle')
    model = build_model()

    print('Injection des poids...')
    inject_weights(model, weights)
    print('Poids injectés avec succès !')

    print('Conversion en TFLite (Full Integer Quantization)...')
    convert_to_tflite(model, out_path)
    print("Modèle exporté sous '%s'" % out_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        p = sys.argv
    else:
        p = 'model.pickle'
    if len(sys.argv) > 2:
        out = sys.argv
    else:
        out = 'mnist_custom.tflite'
    main(p, out)