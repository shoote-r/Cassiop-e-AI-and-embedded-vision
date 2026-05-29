import numpy as np
from sklearn.datasets import fetch_openml
import pickle

# ==========================================
# 1. COUCHE DE CONVOLUTION
# ==========================================
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Initialisation de Xavier (divisé par 9 car 3x3 = 9)
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """Générateur qui extrait des régions 3x3 valides de l'image."""
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input_image):
        """Passe avant : applique les filtres de convolution."""
        self.last_input = input_image
        h, w = input_image.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input_image):
            # Convolution : somme du produit élément par élément
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backprop(self, d_L_d_out, learn_rate):
        """Passe arrière : calcule les gradients des filtres et les met à jour."""
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Le gradient du filtre est la région d'entrée multipliée par le gradient de sortie
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Mise à jour des poids du filtre
        self.filters -= learn_rate * d_L_d_filters
        
        # Note: Pour un réseau plus profond, il faudrait aussi retourner le gradient 
        # par rapport à l'entrée (d_L_d_input), mais ici c'est la première couche.
        return None

# ==========================================
# 2. COUCHE DE MAX POOLING
# ==========================================
class MaxPool2:
    def iterate_regions(self, image):
        """Extrait des blocs non chevauchants de 2x2."""
        h, w, _ = image.shape
        new_h, new_w = h // 2, w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input_volume):
        self.last_input = input_volume
        h, w, num_filters = input_volume.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input_volume):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_L_d_out):
        """Le gradient ne 'coule' qu'à travers le pixel qui était le maximum."""
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # Si ce pixel était le maximum, on lui passe le gradient
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

# ==========================================
# 3. COUCHE DENSE (CLASSIFICATION SOFTMAX)
# ==========================================
class DenseSoftmax:
    def __init__(self, input_len, nodes):
        # Poids et biais pour la classification finale
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input_volume):
        self.last_input_shape = input_volume.shape
        # Aplatir le volume 3D en vecteur 1D
        input_flat = input_volume.flatten()
        self.last_input = input_flat

        # Z = W * X + b
        totals = np.dot(input_flat, self.weights) + self.biases
        
        # Activation Softmax
        exp = np.exp(totals - np.max(totals)) # Soustraction pour stabilité numérique
        self.last_softmax = exp / np.sum(exp, axis=0)
        return self.last_softmax

    def backprop(self, d_L_d_out, learn_rate):
        """
        Calcule les gradients de la perte d'entropie croisée couplée au Softmax.
        d_L_d_out contient le vecteur correct (one-hot).
        """
        # Le gradient combiné Cross-Entropy + Softmax est simplement (Prédictions - Vérité)
        # Mais ici, on passe la VRAIE classe en index (y_true) dans d_L_d_out
        y_true = d_L_d_out
        
        # Initialiser dZ (dérivée de l'erreur par rapport à Z)
        dZ = np.copy(self.last_softmax)
        dZ[y_true] -= 1  # (P - Y)
        
        # Gradients des poids et des biais
        d_L_d_w = np.outer(self.last_input, dZ)
        d_L_d_b = dZ
        
        # Gradient à passer à la couche précédente (MaxPool)
        d_L_d_inputs = np.dot(self.weights, dZ)
        
        # Mise à jour
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b
        
        # Reconstruire la forme 3D pour la couche MaxPool
        return d_L_d_inputs.reshape(self.last_input_shape)

# ==========================================
# 4. ENTRAÎNEMENT ET UTILISATION
# ==========================================

print("Téléchargement de MNIST (cela peut prendre quelques secondes)...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = (mnist.data / 255.0) - 0.5 # Normalisation [-0.5, 0.5]
y = mnist.target.astype(int)

# On sélectionne seulement 1000 images pour cet exemple (car le code Python pur est lent)
train_images = X[:40000].reshape(-1, 28, 28)
train_labels = y[:40000]
test_images = X[40000:60000].reshape(-1, 28, 28)
test_labels = y[40000:60000]

# Initialisation de notre réseau
conv = Conv3x3(8)                 # Entrée 28x28 -> Sortie 26x26x8
pool = MaxPool2()                 # Entrée 26x26x8 -> Sortie 13x13x8
dense = DenseSoftmax(13 * 13 * 8, 10) # 1352 entrées, 10 sorties

def forward(image, label):
    """Passe avant complète et calcul de perte."""
    out = conv.forward((image + 0.5)) # Les images sont centrées sur 0
    out = pool.forward(out)
    out = dense.forward(out)
    
    # Calcul de la perte Cross-Entropy et de l'exactitude
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc

def train(image, label, lr):
    """Entraînement sur une image."""
    # 1. Forward
    out, loss, acc = forward(image, label)

    # 2. Backward
    # Pour notre implémentation Dense, on passe l'index de la vraie classe comme d_L_d_out
    gradient = dense.backprop(label, lr)
    gradient = pool.backprop(gradient)
    conv.backprop(gradient, lr)

    return loss, acc

print("\nDébut de l'entraînement...")
current_lr = 0.005
for epoch in range(50):
    print(f"--- Epoch {epoch + 1} ---")
    
    # Mélange des données
    permutation = np.random.permutation(len(train_images))
    images = train_images[permutation]
    labels = train_labels[permutation]

    loss_sum = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(images, labels)):
        if i > 0 and i % 100 == 0:
            print(f"[Step {i}] Perte moyenne: {loss_sum/100:.3f} | Précision: {num_correct}%")
            loss_sum = 0
            num_correct = 0

        l, acc = train(im, label,current_lr)
        loss_sum += l
        num_correct += acc
    current_lr = current_lr*0.8
# Phase de Test
print("\nTest sur des données non vues...")
loss_sum = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss_sum += l
    num_correct += acc

print(f"Précision sur le jeu de test : {num_correct / len(test_images) * 100}%")


print("\nSauvegarde du modèle en cours...")

# 1. On regroupe les paramètres appris dans un dictionnaire
model_parameters = {
    'conv_filters': conv.filters,
    'dense_weights': dense.weights,
    'dense_biases': dense.biases
}

# 2. On écrit ce dictionnaire dans un fichier .pickle
nom_fichier = 'mon_mini_cnn_numpy.pickle'
with open(nom_fichier, 'wb') as f:
    pickle.dump(model_parameters, f)

print(f"Modèle sauvegardé avec succès sous '{nom_fichier}' !")