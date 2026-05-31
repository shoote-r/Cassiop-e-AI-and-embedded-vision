"""
Primitives de quantification pour la full integer quantization (PTQ).

Deux schémas:
    - SYMÉTRIQUE (pour les poids): zero_point = 0.
      La plage [-a, +a] est mappée sur [-127, 127].
      Avantage: pas de zero_point -> calculs entiers simplifiés.

    - ASYMÉTRIQUE (pour les activations): zero_point quelconque.
      La plage [min, max] est mappée sur [-128, 127].
      Avantage: exploite toute la plage int8 même quand les valeurs
      sont toutes positives (cas typique après une ReLU).

Convention: on quantifie en int8 (8 bits signés, plage [-128, 127]).

Formules:
    quantifier   : q = clip(round(x / scale) + zero_point, -128, 127)
    déquantifier : x = scale * (q - zero_point)
"""

import numpy as np


# Bornes du type int8
QMIN = -128
QMAX = 127


# ============================================================================
# Calcul des paramètres de quantification (scale, zero_point)
# ============================================================================

def compute_symmetric_params(x):   
    """
    Calcule le scale pour une quantification symétrique (zero_point = 0).
    Utilisé pour les poids.

    La plage réelle [-amax, +amax] est mappée sur [-127, +127]
    (on n'utilise pas -128 pour garder la symétrie parfaite).
    """
    amax = np.abs(x).max()
    if amax == 0:
        # Tenseur nul : scale arbitraire non nul pour éviter une division par 0
        return 1.0, 0
    scale = amax / 127.0
    zero_point = 0
    return float(scale), int(zero_point)


def compute_asymmetric_params(x_min, x_max):
    """
    Calcule (scale, zero_point) pour une quantification asymétrique.
    Utilisé pour les activations.

    La plage réelle [x_min, x_max] est mappée sur [QMIN, QMAX].

    Le zero_point est l'entier qui représente la valeur réelle 0.0 :
    c'est important car les zéros (padding, sorties de ReLU) doivent
    être représentés exactement.
    """
    # On s'assure que la plage inclut 0 (sinon le zero_point n'a pas de sens)
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)

    if x_max == x_min:
        return 1.0, 0

    scale = (x_max - x_min) / (QMAX - QMIN)

    # zero_point : on résout q = round(x/scale) + zp avec x=x_min -> q=QMIN
    zero_point = QMIN - round(x_min / scale)
    # On clippe le zero_point dans la plage int8
    zero_point = int(np.clip(zero_point, QMIN, QMAX))

    return float(scale), zero_point


# ============================================================================
# Quantification / déquantification
# ============================================================================

def quantize(x, scale, zero_point):
    """
    Convertit un tenseur float -> int8.

    q = clip(round(x / scale) + zero_point, -128, 127)
    """
    q = np.round(x / scale) + zero_point
    q = np.clip(q, QMIN, QMAX)
    return q.astype(np.int8)


def dequantize(q, scale, zero_point):
    """
    Convertit un tenseur int8 -> float (approximation de l'original).

    x = scale * (q - zero_point)
    """
    return scale * (q.astype(np.float32) - zero_point)


# ============================================================================
# Conteneur pour un tenseur quantifié
# ============================================================================

class QTensor:
    """
    Regroupe un tenseur quantifié et ses paramètres de quantification.

    Stocke:
        q          : les données en int8
        scale      : facteur d'échelle (float)
        zero_point : décalage entier
    """

    def __init__(self, q, scale, zero_point):
        self.q = q
        self.scale = scale
        self.zero_point = zero_point

    @property
    def shape(self):
        return self.q.shape

    @property
    def nbytes(self):
        """Taille mémoire des données quantifiées (les int8 uniquement)."""
        return self.q.nbytes

    def dequantize(self):
        """Reconstruit l'approximation float du tenseur."""
        return dequantize(self.q, self.scale, self.zero_point)

    @classmethod
    def from_float_symmetric(cls, x):
        """Quantifie un tenseur float en symétrique (pour les poids)."""
        scale, zp = compute_symmetric_params(x)
        q = quantize(x, scale, zp)
        return cls(q, scale, zp)

    @classmethod
    def from_float_asymmetric(cls, x, x_min=None, x_max=None):
        """Quantifie un tenseur float en asymétrique (pour les activations)."""
        if x_min is None:
            x_min = float(x.min())
        if x_max is None:
            x_max = float(x.max())
        scale, zp = compute_asymmetric_params(x_min, x_max)
        q = quantize(x, scale, zp)
        return cls(q, scale, zp)

    def __repr__(self):
        return (f"QTensor(shape={self.shape}, scale={self.scale:.6g}, "
                f"zero_point={self.zero_point})")


# ============================================================================
# Mesure de l'erreur de quantification
# ============================================================================

def quantization_error(x_float, qtensor):
    """
    Mesure l'erreur introduite par la quantification.

    Retourne un dict avec plusieurs métriques d'erreur.
    """
    x_reconstructed = qtensor.dequantize()
    abs_err = np.abs(x_float - x_reconstructed)

    # SQNR : Signal-to-Quantization-Noise Ratio (en dB)
    # Plus c'est élevé, mieux c'est. > 20 dB est généralement correct.
    # SQNR : rapport entre la puissance du signal et la puissance de l'erreur d'arrondi introduite 
    signal_power = np.mean(x_float ** 2)
    noise_power = np.mean(abs_err ** 2)
    if noise_power == 0:
        sqnr_db = float('inf')                   
    else:
        sqnr_db = 10 * np.log10(signal_power / noise_power)

    return {
        'max_abs_error': float(abs_err.max()),
        'mean_abs_error': float(abs_err.mean()),
        'sqnr_db': float(sqnr_db),
    }


if __name__ == '__main__':
    # Test des primitives sur des données synthétiques
    np.random.seed(0)

    print("=== Test quantification symétrique (poids) ===")
    w = np.random.randn(3, 3, 8, 16).astype(np.float32) * 0.5
    qw = QTensor.from_float_symmetric(w)
    print(qw)
    err = quantization_error(w, qw)
    print(f"  erreur max={err['max_abs_error']:.5f}, "
          f"moyenne={err['mean_abs_error']:.5f}, SQNR={err['sqnr_db']:.1f} dB")

    print("\n=== Test quantification asymétrique (activations post-ReLU) ===")
    # Activations après ReLU : toutes >= 0
    a = np.abs(np.random.randn(64, 14, 14, 8).astype(np.float32)) * 2.0
    qa = QTensor.from_float_asymmetric(a)
    print(qa)
    err = quantization_error(a, qa)
    print(f"  erreur max={err['max_abs_error']:.5f}, "
          f"moyenne={err['mean_abs_error']:.5f}, SQNR={err['sqnr_db']:.1f} dB")

    print("\n=== Vérification : zero_point représente bien 0.0 ===")
    zero_q = quantize(np.array([0.0], dtype=np.float32), qa.scale, qa.zero_point)
    print(f"  quantize(0.0) = {zero_q[0]} (doit valoir zero_point={qa.zero_point})")