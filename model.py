"""
Modèle LeNet simplifié pour MNIST.

Architecture:
    Input (28, 28, 1)
      -> Conv 3x3, 1->8, pad=1   (sortie 28x28x8)
      -> ReLU
      -> MaxPool 2x2             (sortie 14x14x8)
      -> Conv 3x3, 8->16, pad=1  (sortie 14x14x16)
      -> ReLU
      -> MaxPool 2x2             (sortie 7x7x16)
      -> Flatten                 (sortie 784)
      -> Dense 784 -> 64
      -> ReLU
      -> Dense 64 -> 10
      -> SoftmaxCrossEntropy (loss)

Total ~52k paramètres.
"""

import numpy as np
from layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy


class LeNet:

    def __init__(self):
        # Bloc convolutif 1
        self.conv1 = Conv2D(in_channels=1,  out_channels=8,  kernel_size=3, stride=1, pad=1, name='conv1')
        self.relu1 = ReLU(name='relu1')
        self.pool1 = MaxPool2D(pool_size=2, stride=2, name='pool1')

        # Bloc convolutif 2
        self.conv2 = Conv2D(in_channels=8,  out_channels=16, kernel_size=3, stride=1, pad=1, name='conv2')
        self.relu2 = ReLU(name='relu2')
        self.pool2 = MaxPool2D(pool_size=2, stride=2, name='pool2')

        # Tête classification
        self.flatten = Flatten(name='flatten')
        self.fc1   = Dense(in_features=7*7*16, out_features=64, name='fc1')
        self.relu3 = ReLU(name='relu3')
        self.fc2   = Dense(in_features=64, out_features=10, name='fc2')

        # Loss
        self.criterion = SoftmaxCrossEntropy()

        # Liste ordonnée des couches (pour forward/backward automatiques)
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3,
            self.fc2,
        ]

    # ------------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------------

    def forward(self, x):
        """Forward sans loss. Retourne les logits."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss_and_backward(self, x, y):
        """Forward complet + backward complet. Retourne (loss, probas)."""
        logits = self.forward(x)
        loss, probs = self.criterion.forward(logits, y)

        # Rétropropagation
        dout = self.criterion.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return loss, probs

    def predict(self, x):
        """Inférence pure. Retourne les indices de classe prédits."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x):
        """
        Inférence retournant les probabilités softmax (N, 10).
        Utile pour afficher une vraie confiance en pourcentage.
        """
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------------
    # Accès aux paramètres (pour l'optimiseur)
    # ------------------------------------------------------------------------

    def parameters(self):
        """Itère sur tous les (param_array, grad_array) à mettre à jour."""
        for layer in self.layers:
            for name, p, g in layer.params():
                yield p, g

    def num_params(self):
        return sum(p.size for p, _ in self.parameters())

    # ------------------------------------------------------------------------
    # Sauvegarde / chargement (NumPy .npz, pas de pickle)
    # ------------------------------------------------------------------------

    def save(self, path):
        """Sauvegarde tous les poids dans un fichier .npz."""
        data = {}
        for layer in self.layers:
            for name, p, _ in layer.params():
                key = f"{layer.name}_{name}"
                data[key] = p
        np.savez(path, **data)
        print(f"Modèle sauvegardé dans {path}")

    def load(self, path):
        """Charge les poids depuis un fichier .npz."""
        data = np.load(path)
        for layer in self.layers:
            for name, p, _ in layer.params():
                key = f"{layer.name}_{name}"
                if key in data:
                    p[...] = data[key]  # copie in-place pour préserver les références
                else:
                    print(f"  ⚠ clé manquante: {key}")
        print(f"Modèle chargé depuis {path}")


if __name__ == '__main__':
    np.random.seed(0)
    model = LeNet()
    print(f"Nombre total de paramètres: {model.num_params():,}")

    # Test forward / backward sur un mini-batch aléatoire
    x = np.random.rand(4, 28, 28, 1).astype(np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int32)

    loss, probs = model.loss_and_backward(x, y)
    print(f"Loss initiale: {loss:.4f}  (attendu ~{np.log(10):.4f} pour 10 classes équiprobables)")
    print(f"Probs shape: {probs.shape}, somme par ligne: {probs.sum(axis=1)}")

    # Vérification que les gradients ont bien été calculés
    for layer in model.layers:
        for name, p, g in layer.params():
            print(f"  {layer.name}.{name}: param {p.shape}, grad norm {np.linalg.norm(g):.4f}")