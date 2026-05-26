"""
Couches du réseau de neurones, en NumPy pur mais vectorisées.

Conventions:
    - Les couches sont des CLASSES avec un état interne (poids + cache).
      Le cache stocke les valeurs nécessaires à back-propagation
    - forward(x) -> y
    - backward(dy) -> dx (et stocke les gradients des poids dans self.grads)

Note sur im2col:
    im2col transforme la conv en GEMM (multiplication matricielle géante)
    que NumPy/BLAS exécute en code natif optimisé.
"""

import numpy as np

def im2col(x, kernel_h, kernel_w, stride=1, pad=0):
    """
    Transforme un batch d'images (N, H, W, C) en une matrice 2D où
    chaque colonne contient une "fenêtre" aplatie sur laquelle on appliquera
    le filtre. Permet de remplacer la conv par un np.dot().

    Args:
        x: (N, H, W, C) un tenseur 
    Returns:
        cols: (N * H_out * W_out, kernel_h * kernel_w * C)
        out_shape: (H_out, W_out) pour reconstruire ensuite
    """
    N, H, W, C = x.shape
    H_out = (H + 2*pad - kernel_h) // stride + 1
    W_out = (W + 2*pad - kernel_w) // stride + 1

    # Padding avec des zéros sur les dimensions spatiales uniquement
    if pad > 0:
        x_padded = np.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant')
    else:
        x_padded = x

    # On construit cols via une vue "strided" pour éviter une boucle Python.
    # Idée: créer un tenseur (N, H_out, W_out, kernel_h, kernel_w, C) où
    # chaque (i,j) contient la fenêtre extraite à cette position.
    cols = np.zeros((N, H_out, W_out, kernel_h, kernel_w, C), dtype=x.dtype)
    for i in range(kernel_h):
        i_max = i + stride * H_out
        for j in range(kernel_w):
            j_max = j + stride * W_out
            cols[:, :, :, i, j, :] = x_padded[:, i:i_max:stride, j:j_max:stride, :]

    # Reshape final: (N * H_out * W_out, kh * kw * C)
    cols = cols.reshape(N * H_out * W_out, kernel_h * kernel_w * C)
    return cols, (H_out, W_out)


def col2im(cols, x_shape, kernel_h, kernel_w, stride=1, pad=0):
    """
    Inverse de im2col pour la passe arrière de la convolution.
    Comme une cellule de l'entrée peut contribuer à plusieurs colonnes,
    on ACCUMULE les gradients (d'où le += dans la boucle).

    Args:
        cols: (N * H_out * W_out, kernel_h * kernel_w * C)
        x_shape: forme originale (N, H, W, C)
    Returns:
        dx: (N, H, W, C)
    """
    N, H, W, C = x_shape
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    H_out = (H_pad - kernel_h) // stride + 1
    W_out = (W_pad - kernel_w) // stride + 1

    cols = cols.reshape(N, H_out, W_out, kernel_h, kernel_w, C)
    x_padded = np.zeros((N, H_pad, W_pad, C), dtype=cols.dtype)

    for i in range(kernel_h):
        i_max = i + stride * H_out
        for j in range(kernel_w):
            j_max = j + stride * W_out
            x_padded[:, i:i_max:stride, j:j_max:stride, :] += cols[:, :, :, i, j, :]

    if pad == 0:
        return x_padded
    return x_padded[:, pad:-pad, pad:-pad, :]


# ============================================================================
# Couche de convolution 2D
# ============================================================================

class Conv2D:
    """
    Convolution 2D avec padding et stride configurables.

    Initialisation des poids : He (adaptée à ReLU).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, name='conv'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.name = name

        # Poids : (kernel_h, kernel_w, in_channels, out_channels)
        fan_in = kernel_size * kernel_size * in_channels
        self.W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels).astype(np.float32)
        self.W *= np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels, dtype=np.float32)

        # Gradients (remplis pendant backward, lus par l'optimiseur)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache = None

    def forward(self, x):
        """x: (N, H, W, C_in) -> out: (N, H_out, W_out, C_out)"""
        N = x.shape[0]
        cols, (H_out, W_out) = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)

        # W aplati : (kh * kw * C_in, C_out)
        W_flat = self.W.reshape(-1, self.out_channels)

        # cols @ W_flat : (N * H_out * W_out, C_out)
        out = cols @ W_flat + self.b
        out = out.reshape(N, H_out, W_out, self.out_channels)

        self.cache = (x, cols, W_flat)
        return out

    def backward(self, dout):
        """dout: (N, H_out, W_out, C_out) -> dx: (N, H, W, C_in)"""
        x, cols, W_flat = self.cache
        N = x.shape[0]

        # dout aplati : (N * H_out * W_out, C_out)
        dout_flat = dout.reshape(-1, self.out_channels)

        # Gradient des biais : somme sur toutes les positions et tout le batch
        self.db = dout_flat.sum(axis=0)

        # Gradient des poids : cols.T @ dout_flat
        dW_flat = cols.T @ dout_flat  # (kh * kw * C_in, C_out)
        self.dW = dW_flat.reshape(self.W.shape)

        # Gradient de l'entrée : dout_flat @ W_flat.T, puis col2im
        dcols = dout_flat @ W_flat.T
        dx = col2im(dcols, x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        return dx

    def params(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]


# ============================================================================
# Max Pooling 2x2
# ============================================================================

class MaxPool2D:
    """
    MaxPooling 2D. Garde le maximum sur chaque fenêtre.
    Le gradient ne passe que par les positions du maximum (les autres reçoivent 0).
    """

    def __init__(self, pool_size=2, stride=2, name='pool'):
        self.pool_size = pool_size
        self.stride = stride
        self.name = name
        self.cache = None

    def forward(self, x):
        """x: (N, H, W, C) -> out: (N, H_out, W_out, C)"""
        N, H, W, C = x.shape
        ps = self.pool_size
        s = self.stride
        H_out = (H - ps) // s + 1
        W_out = (W - ps) // s + 1

        # Astuce : on fait im2col canal par canal pour garder C séparé.
        # Plus simple : on reshape pour exposer les fenêtres.
        # x[:, i::s, j::s, :] donne tous les pixels en position (i,j) de chaque fenêtre.
        # On empile les ps*ps positions et on prend le max.
        windows = np.zeros((N, H_out, W_out, ps * ps, C), dtype=x.dtype)
        for i in range(ps):
            for j in range(ps):
                windows[:, :, :, i*ps + j, :] = x[:, i:i+s*H_out:s, j:j+s*W_out:s, :]

        # argmax pour savoir d'où vient chaque max (utilisé en backward)
        argmax = np.argmax(windows, axis=3)  # (N, H_out, W_out, C)
        out = np.max(windows, axis=3)        # (N, H_out, W_out, C)

        self.cache = (x.shape, argmax)
        return out

    def backward(self, dout):
        """dout: (N, H_out, W_out, C) -> dx: (N, H, W, C)"""
        x_shape, argmax = self.cache
        N, H, W, C = x_shape
        ps = self.pool_size
        s = self.stride
        H_out, W_out = dout.shape[1], dout.shape[2]

        dx = np.zeros(x_shape, dtype=dout.dtype)

        # Pour chaque position de la sortie, on route le gradient vers la
        # position du max dans l'entrée.
        for i in range(ps):
            for j in range(ps):
                flat_idx = i * ps + j
                mask = (argmax == flat_idx)  # (N, H_out, W_out, C)
                dx[:, i:i+s*H_out:s, j:j+s*W_out:s, :] += dout * mask

        return dx

    def params(self):
        return []


# ============================================================================
# Flatten
# ============================================================================

class Flatten:
    """Aplatit (N, H, W, C) en (N, H*W*C)."""

    def __init__(self, name='flatten'):
        self.name = name
        self.cache = None

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)

    def params(self):
        return []


# ============================================================================
# Couche dense
# ============================================================================

class Dense:
    """Couche entièrement connectée: y = x @ W + b"""

    def __init__(self, in_features, out_features, name='dense'):
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        # He init
        self.W = np.random.randn(in_features, out_features).astype(np.float32)
        self.W *= np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache = None

    def forward(self, x):
        """x: (N, in_features) -> (N, out_features)"""
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        """dout: (N, out_features) -> dx: (N, in_features)"""
        x = self.cache
        self.dW = x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

    def params(self):
        return [('W', self.W, self.dW), ('b', self.b, self.db)]


# ============================================================================
# Activations
# ============================================================================

class ReLU:
    def __init__(self, name='relu'):
        self.name = name
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.cache > 0)

    def params(self):
        return []


# ============================================================================
# Softmax + Cross-Entropy (fusionnés pour stabilité numérique et simplicité)
# ============================================================================

class SoftmaxCrossEntropy:
    """
    Combine softmax et la cross-entropy en une seule "tête" de modèle.

    Pourquoi fusionner ?
    1. Stabilité numérique : softmax(x) suivi de log() risque de produire
       log(0) -> -inf. En fusionnant, on simplifie mathématiquement.
    2. Le gradient combiné est ÉLÉGANT : dL/dz = (predictions - one_hot(target))
       au lieu d'un produit Jacobien horrible.
    """

    def __init__(self):
        self.cache = None

    def forward(self, logits, targets):
        """
        logits: (N, K) sorties brutes du réseau
        targets: (N,) labels entiers dans [0, K-1]
        Returns: (loss scalaire, probs (N, K))
        """
        # Softmax stable : on soustrait le max ligne par ligne
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        N = logits.shape[0]
        # Cross-entropy : -log(p_target) moyenné sur le batch
        # On ajoute un epsilon pour éviter log(0) en pratique
        log_probs = -np.log(probs[np.arange(N), targets] + 1e-12)
        loss = log_probs.mean()

        self.cache = (probs, targets)
        return loss, probs

    def backward(self):
        """Retourne dL/dlogits, à propager dans la dernière couche dense."""
        probs, targets = self.cache
        N = probs.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(N), targets] -= 1.0
        dlogits /= N  # moyenne sur le batch (cohérent avec loss.mean())
        return dlogits