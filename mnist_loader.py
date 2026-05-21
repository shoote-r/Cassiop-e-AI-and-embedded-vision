"""
Chargement du dataset MNIST.

Télécharge automatiquement les 4 fichiers IDX gzippés si absents,
puis les parse manuellement. Aucune dépendance à Keras/torchvision.

Format IDX (cf. http://yann.lecun.com/exdb/mnist/):
    - 4 bytes : magic number
    - 4 bytes : nombre d'éléments
    - (pour les images) 4 bytes : nb lignes, 4 bytes : nb colonnes
    - N bytes : données (unsigned bytes, big-endian)
"""

import os
import gzip
import struct
import urllib.request
import numpy as np


# Miroirs MNIST. Le site officiel de Yann LeCun est souvent indisponible,
# donc on utilise un miroir maintenu par ossci-datasets (PyTorch).
MNIST_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images':  't10k-images-idx3-ubyte.gz',
    'test_labels':  't10k-labels-idx1-ubyte.gz',
}


def _download_if_missing(filename, target_dir):
    """Télécharge un fichier MNIST si absent du disque."""
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    url = MNIST_URL_BASE + filename
    print(f"  Téléchargement de {filename}...")
    urllib.request.urlretrieve(url, target_path)
    return target_path


def _parse_idx_images(path):
    """Parse un fichier IDX d'images et retourne un array (N, H, W) en uint8."""
    with gzip.open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number invalide pour les images: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def _parse_idx_labels(path):
    """Parse un fichier IDX de labels et retourne un array (N,) en uint8."""
    with gzip.open(path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number invalide pour les labels: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(data_dir='mnist_data', normalize=True, max_train=None, max_test=None):
    """
    Charge MNIST en mémoire.

    Returns:
        X_train: (N_train, 28, 28, 1) float32 dans [0,1] si normalize=True
        Y_train: (N_train,) int32
        X_test:  (N_test, 28, 28, 1) float32
        Y_test:  (N_test,) int32
    """
    print("Chargement de MNIST...")

    paths = {
        key: _download_if_missing(fname, data_dir)
        for key, fname in FILES.items()
    }

    X_train = _parse_idx_images(paths['train_images'])
    Y_train = _parse_idx_labels(paths['train_labels'])
    X_test  = _parse_idx_images(paths['test_images'])
    Y_test  = _parse_idx_labels(paths['test_labels'])

    # Limitation optionnelle (utile pour itérer rapidement pendant le dev)
    if max_train is not None:
        X_train = X_train[:max_train]
        Y_train = Y_train[:max_train]
    if max_test is not None:
        X_test = X_test[:max_test]
        Y_test = Y_test[:max_test]

    # Ajout de la dimension canal et conversion en float32
    X_train = X_train.reshape(-1, 28, 28, 1).astype(np.float32)
    X_test  = X_test.reshape(-1, 28, 28, 1).astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test  /= 255.0

    Y_train = Y_train.astype(np.int32)
    Y_test  = Y_test.astype(np.int32)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    # Sanity check
    Xtr, Ytr, Xte, Yte = load_mnist()
    print(f"Plage des valeurs train: [{Xtr.min():.3f}, {Xtr.max():.3f}]")
    print(f"Classes train: {np.bincount(Ytr)}")
    print(f"Classes test:  {np.bincount(Yte)}")