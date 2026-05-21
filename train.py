"""
Boucle d'entraînement du LeNet sur MNIST.

Choix techniques:
    - Optimiseur: SGD avec momentum (simple, robuste, peu d'hyperparamètres)
    - Batch size: 64 (bon compromis vitesse/qualité du gradient)
    - LR: 0.01 avec decay multiplicatif après chaque epoch
    - Augmentation: translations, rotations légères, bruit, variations d'épaisseur
      (cruciales pour la robustesse webcam)
"""

import time
import numpy as np

from mnist_loader import load_mnist
from model import LeNet


# ============================================================================
# Augmentation de données — orientée robustesse webcam
# ============================================================================

def random_shift(img, max_shift=2):
    """Translation avec padding noir (pas de wrap-around)."""
    sx = np.random.randint(-max_shift, max_shift + 1)
    sy = np.random.randint(-max_shift, max_shift + 1)
    out = np.zeros_like(img)
    H, W = img.shape[:2]

    # Sources et destinations en x
    src_x0 = max(0, -sx);  src_x1 = W - max(0, sx)
    dst_x0 = max(0, sx);   dst_x1 = W - max(0, -sx)
    # Sources et destinations en y
    src_y0 = max(0, -sy);  src_y1 = H - max(0, sy)
    dst_y0 = max(0, sy);   dst_y1 = H - max(0, -sy)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def random_rotation(img, max_deg=10):
    """Rotation par interpolation bilinéaire. Simple, pas de dépendance externe."""
    angle = np.random.uniform(-max_deg, max_deg)
    if abs(angle) < 0.5:
        return img

    H, W, C = img.shape
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = W / 2, H / 2

    # Grille de coordonnées destination, on remonte aux sources
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = cos_t * (xx - cx) + sin_t * (yy - cy) + cx
    ys = -sin_t * (xx - cx) + cos_t * (yy - cy) + cy

    # Bilinéaire
    x0 = np.floor(xs).astype(int);  x1 = x0 + 1
    y0 = np.floor(ys).astype(int);  y1 = y0 + 1

    # Pondérations
    wx1 = xs - x0;  wx0 = 1 - wx1
    wy1 = ys - y0;  wy0 = 1 - wy1

    # Clip pour rester dans les bornes (les pixels hors-cadre seront noirs)
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    x0 = np.clip(x0, 0, W - 1); x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1); y1 = np.clip(y1, 0, H - 1)

    out = (img[y0, x0] * (wy0 * wx0)[..., None] +
           img[y0, x1] * (wy0 * wx1)[..., None] +
           img[y1, x0] * (wy1 * wx0)[..., None] +
           img[y1, x1] * (wy1 * wx1)[..., None])
    out = out * valid[..., None]  # zeros hors-cadre
    return out.astype(img.dtype)


def random_thickness(img):
    """Variation d'épaisseur de trait (dilatation/érosion par max/min pooling 3x3)."""
    if np.random.rand() > 0.5:
        return img
    H, W, C = img.shape
    # Padding pour le pooling 3x3
    padded = np.pad(img, ((1,1), (1,1), (0,0)), mode='constant')
    # On empile les 9 versions décalées
    stack = np.zeros((9, H, W, C), dtype=img.dtype)
    idx = 0
    for di in range(3):
        for dj in range(3):
            stack[idx] = padded[di:di+H, dj:dj+W, :]
            idx += 1
    if np.random.rand() > 0.5:
        return stack.max(axis=0)  # dilatation: trait plus épais
    else:
        return stack.min(axis=0)  # érosion: trait plus fin


def augment_batch(X):
    """Augmente un batch entier (loop Python — coût ~négligeable face à la conv)."""
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        img = X[i]
        if np.random.rand() > 0.3:
            img = random_shift(img, max_shift=2)
        if np.random.rand() > 0.5:
            img = random_rotation(img, max_deg=10)
        img = random_thickness(img)
        if np.random.rand() > 0.5:
            img = img + np.random.randn(*img.shape).astype(np.float32) * 0.05
        img = np.clip(img, 0.0, 1.0)
        out[i] = img
    return out


# ============================================================================
# Optimiseur SGD avec momentum
# ============================================================================

class SGDMomentum:
    """
    Mise à jour des poids avec momentum:
        v = momentum * v - lr * grad
        p = p + v

    IMPORTANT: on garde des références aux LAYERS (pas aux arrays de gradients)
    car les backward() de nos couches RÉASSIGNENT self.dW à de nouveaux arrays
    à chaque appel. Une référence directe à l'ancien array serait obsolète.
    """

    def __init__(self, model, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.model = model

        # On indexe par (layer, nom_du_param) pour récupérer la velocité
        # ET on met à jour le param via setattr sur le layer.
        self.velocities = {}
        for layer in model.layers:
            for name, p, _g in layer.params():
                # Note: on stocke la velocity côté optimiseur, pas côté layer
                self.velocities[(id(layer), name)] = np.zeros_like(p)

    def step(self):
        for layer in self.model.layers:
            for name, p, g in layer.params():
                v = self.velocities[(id(layer), name)]
                v *= self.momentum
                v -= self.lr * g
                p += v  # update in-place du tenseur de poids (qui lui n'est pas réassigné)


# ============================================================================
# Évaluation
# ============================================================================

def evaluate(model, X, Y, batch_size=128):
    """Calcule la précision (accuracy) sur tout un dataset."""
    correct = 0
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        yb = Y[i:i+batch_size]
        preds = model.predict(xb)
        correct += (preds == yb).sum()
    return correct / N


# ============================================================================
# Boucle d'entraînement principale
# ============================================================================

def train(epochs=10, batch_size=64, lr=0.01, lr_decay=0.9,
          max_train=30000, max_test=5000, seed=42, save_path='lenet_mnist.npz'):

    np.random.seed(seed)

    # Données
    X_train, Y_train, X_test, Y_test = load_mnist(
        max_train=max_train, max_test=max_test
    )
    N = X_train.shape[0]
    print(f"Entraînement sur {N} échantillons, test sur {X_test.shape[0]}")

    # Modèle + optimiseur
    model = LeNet()
    print(f"Paramètres totaux: {model.num_params():,}")
    optimizer = SGDMomentum(model, lr=lr, momentum=0.9)

    # Suivi
    best_test_acc = 0.0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Shuffle des indices d'entraînement
        perm = np.random.permutation(N)

        # Statistiques d'epoch
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for batch_start in range(0, N, batch_size):
            idx = perm[batch_start:batch_start+batch_size]
            xb = X_train[idx]
            yb = Y_train[idx]

            # Augmentation à la volée (uniquement pour le train)
            xb = augment_batch(xb)

            # Forward + Backward
            loss, probs = model.loss_and_backward(xb, yb)

            # Update
            optimizer.step()

            epoch_loss += loss
            epoch_correct += (np.argmax(probs, axis=1) == yb).sum()
            n_batches += 1

            # Affichage périodique
            if n_batches % 50 == 0:
                avg_loss = epoch_loss / n_batches
                acc_so_far = epoch_correct / (n_batches * batch_size)
                print(f"  Epoch {epoch} | batch {n_batches}/{N // batch_size} | "
                      f"loss {avg_loss:.4f} | train_acc {acc_so_far*100:.2f}%")

        epoch_time = time.time() - epoch_start

        # Évaluation test à chaque epoch
        test_acc = evaluate(model, X_test, Y_test)

        train_loss = epoch_loss / n_batches
        train_acc = epoch_correct / (n_batches * batch_size)

        print(f"[Epoch {epoch}/{epochs}] time={epoch_time:.1f}s | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"test_acc={test_acc*100:.2f}% | lr={optimizer.lr:.4f}")

        # Sauvegarde du meilleur modèle
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(save_path)

        # LR decay
        optimizer.lr *= lr_decay

    total_time = time.time() - train_start
    print(f"\nEntraînement terminé en {total_time/60:.1f} min")
    print(f"Meilleure précision test: {best_test_acc*100:.2f}%")
    return model


if __name__ == '__main__':
    train(
        epochs=15,
        batch_size=64,
        lr=0.01,
        lr_decay=0.92,
        max_train=60000,
        max_test=10000,
    )