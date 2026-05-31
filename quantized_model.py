"""
Calibration et modèle quantifié full-integer.

Étapes:
    1. CALIBRATION : on fait passer un échantillon de données dans le modèle
       float et on enregistre, pour chaque couche, la plage [min, max] de
       ses activations de sortie.
    2. QUANTIFICATION : on quantifie les poids (symétrique) et on prépare
       les (scale, zero_point) des activations à partir de la calibration.
    3. FORWARD QUANTIFIÉ : inférence en arithmétique entière.

Ce module NE MODIFIE PAS model.py / layers.py : il lit le modèle float
de l'extérieur via sa liste publique `model.layers`.
"""

import numpy as np

from layers import Conv2D, Dense, ReLU, MaxPool2D, Flatten
from quantization import (
    QTensor, quantize, dequantize,
    compute_symmetric_params, compute_asymmetric_params,
)


# ============================================================================
# Étape 1 : Calibration
# ============================================================================

def collect_activation_ranges(model, X_calib):
    """
    Fait passer les données de calibration couche par couche et enregistre
    la plage [min, max] des activations.

    Args:
        model: LeNet float entraîné
        X_calib: (N, 28, 28, 1) échantillon représentatif

    Returns:
        ranges: liste de dicts {min, max}, un par couche + l'entrée
                ranges[0]   = plage de l'entrée du réseau
                ranges[i+1] = plage de sortie de model.layers[i]
    """
    print(f"Calibration sur {len(X_calib)} échantillons...")

    ranges = []

    # Plage de l'entrée du réseau
    x = X_calib
    ranges.append({'min': float(x.min()), 'max': float(x.max())})

    # On propage couche par couche en enregistrant chaque sortie
    for layer in model.layers:
        x = layer.forward(x)
        ranges.append({'min': float(x.min()), 'max': float(x.max())})

    for i, r in enumerate(ranges):  # enumerate c'est comme une boucle
        tag = 'input' if i == 0 else model.layers[i-1].name
        print(f"  [{tag:10s}] min={r['min']:8.4f}  max={r['max']:8.4f}")

    return ranges


# ============================================================================
# Étape 2 : Modèle quantifié
# ============================================================================

class QuantizedLeNet:
    """
    Version quantifiée full-integer du LeNet.

    Contient:
        - les poids quantifiés (int8, symétrique) de chaque conv/dense
        - les (scale, zero_point) des activations issus de la calibration

    Le forward s'exécute en arithmétique entière (accumulation int32).
    """

    def __init__(self, float_model, activation_ranges):
        self.float_model = float_model
        self.layers = float_model.layers
        self.act_ranges = activation_ranges

        # Pré-calcule les (scale, zp) des activations pour chaque "point"
        # du réseau (entrée + sortie de chaque couche).
        self.act_scales = []
        self.act_zps = []
        for r in activation_ranges:
            scale, zp = compute_asymmetric_params(r['min'], r['max'])
            self.act_scales.append(scale)
            self.act_zps.append(zp)

        # Quantifie les poids des couches conv et dense
        self.qweights = {}   # name -> QTensor (poids)
        self.qbias = {}      # name -> bias quantifié en int32
        self._quantize_weights()

    def _quantize_weights(self):
        """
        Quantifie les poids (symétrique int8) et les biais (int32).

        Le biais est quantifié avec le scale combiné s_x * s_W car il
        s'ajoute au résultat de l'accumulation entière, qui est dans
        cette échelle-là.
        """
        for idx, layer in enumerate(self.layers):  #idx est l'index du enumerate 
            if not isinstance(layer, (Conv2D, Dense)):
                continue

            # Poids : quantification symétrique
            qW = QTensor.from_float_symmetric(layer.W)
            self.qweights[layer.name] = qW

            # Scale de l'entrée de cette couche = scale d'activation à l'index idx
            # (ranges[idx] est la sortie de layers[idx-1] = entrée de layers[idx])
            s_x = self.act_scales[idx]
            s_w = qW.scale
            s_bias = s_x * s_w

            # Biais quantifié en int32 (pas int8 : il s'ajoute aux accumulateurs)
            q_bias = np.round(layer.b / s_bias).astype(np.int32)
            self.qbias[layer.name] = q_bias

    # ------------------------------------------------------------------------
    # Forward quantifié
    # ------------------------------------------------------------------------

    def forward(self, x_float):
        """
        Inférence full-integer.

        Pour rester lisible, on quantifie l'entrée, puis chaque couche
        travaille en entier. Entre deux couches on "requantifie" :
        on repasse par du float le temps d'ajuster scale/zero_point.

        NOTE: une vraie implémentation embarquée éviterait tout retour au
        float en propageant des "multiplicateurs entiers". Ici on garde
        une version lisible et mesurable : les multiplications principales
        (conv, dense) se font bien en int, ce qui suffit pour mesurer
        l'impact mémoire et servir de proxy.

        Returns: logits (N, 10) en float
        """
        # Quantifie l'entrée avec les params de calibration
        s_in = self.act_scales[0]
        zp_in = self.act_zps[0]
        q_x = quantize(x_float, s_in, zp_in)

        for idx, layer in enumerate(self.layers):
            s_in = self.act_scales[idx]
            zp_in = self.act_zps[idx]
            s_out = self.act_scales[idx + 1]
            zp_out = self.act_zps[idx + 1]

            if isinstance(layer, (Conv2D, Dense)):
                q_x = self._forward_linear(layer, q_x, s_in, zp_in, s_out, zp_out)
            elif isinstance(layer, ReLU):
                q_x = self._forward_relu(q_x, zp_in, s_in, zp_in, s_out, zp_out)
            elif isinstance(layer, (MaxPool2D, Flatten)):
                # MaxPool et Flatten ne changent pas les valeurs -> on
                # peut opérer directement sur les entiers via le forward float.
                # (le max d'entiers = max des floats correspondants)
                q_x = self._forward_passthrough(layer, q_x, s_in, zp_in, s_out, zp_out)

        # Dernière sortie : on déquantifie en float pour obtenir les logits
        return dequantize(q_x, self.act_scales[-1], self.act_zps[-1])

    def _forward_linear(self, layer, q_x, s_in, zp_in, s_out, zp_out):
        """
        Cœur du calcul entier pour Conv2D et Dense.

        y_float = s_in*s_w * [ (q_x - zp_in) (conv/matmul) q_w ] + bias
        puis on re-quantifie en (s_out, zp_out).
        """
        qW = self.qweights[layer.name]
        q_bias = self.qbias[layer.name]

        # Accumulation entière. On caste en int32 pour éviter tout overflow.
        x_int = q_x.astype(np.int32) - zp_in   # poids symétriques: zp_w = 0
        w_int = qW.q.astype(np.int32)

        if isinstance(layer, Dense):
            acc = x_int @ w_int                 # (N, out), int32
        else:
            # Conv : on réutilise im2col mais en arithmétique entière
            acc = self._conv_int(layer, x_int, w_int)

        acc = acc + q_bias                      # biais en int32

        # Retour au float réel puis requantification pour la couche suivante
        s_bias = s_in * qW.scale
        y_float = acc.astype(np.float32) * s_bias
        q_out = quantize(y_float, s_out, zp_out)
        return q_out

    def _conv_int(self, layer, x_int, w_int):
        """Convolution en entier via im2col (accumulation int32)."""
        from layers import im2col
        N = x_int.shape[0]
        cols, (H_out, W_out) = im2col(
            x_int, layer.kernel_size, layer.kernel_size, layer.stride, layer.pad
        )
        # cols est int (im2col préserve le dtype) ; w aplati
        w_flat = w_int.reshape(-1, layer.out_channels)
        acc = cols.astype(np.int32) @ w_flat
        return acc.reshape(N, H_out, W_out, layer.out_channels)

    def _forward_relu(self, q_x, zp_x, s_in, zp_in, s_out, zp_out):
        """
        ReLU en quantifié : mettre à zéro tout ce qui est sous la valeur
        réelle 0, ce qui correspond à l'entier zp_x.

        Comme l'entrée et la sortie d'une ReLU ont des plages différentes,
        on requantifie proprement.
        """
        # ReLU : max(0_réel, x). 0_réel correspond à l'entier zp_x.
        x_clamped = np.maximum(q_x.astype(np.int32), zp_x)
        # Déquantifie puis requantifie vers (s_out, zp_out)
        x_float = s_in * (x_clamped - zp_in)
        return quantize(x_float, s_out, zp_out)

    def _forward_passthrough(self, layer, q_x, s_in, zp_in, s_out, zp_out):
        """
        MaxPool / Flatten : opérations qui ne créent pas de nouvelles
        valeurs (juste sélection/réarrangement). On les applique sur les
        entiers directement, puis on requantifie (s_in/s_out diffèrent
        rarement ici mais on reste cohérent).
        """
        out_int = layer.forward(q_x.astype(np.float32))
        x_float = s_in * (out_int - zp_in)
        return quantize(x_float, s_out, zp_out)

    def predict(self, x_float):
        logits = self.forward(x_float)
        return np.argmax(logits, axis=1)

    # ------------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------------

    def save(self, path):
        """Sauvegarde le modèle quantifié dans un .npz (int8 + métadonnées)."""
        data = {}
        for name, qt in self.qweights.items():
            data[f"{name}_qW"] = qt.q
            data[f"{name}_scale"] = np.float32(qt.scale)
        for name, qb in self.qbias.items():
            data[f"{name}_qbias"] = qb
        data['act_scales'] = np.array(self.act_scales, dtype=np.float32)
        data['act_zps'] = np.array(self.act_zps, dtype=np.int32)
        np.savez(path, **data)
        print(f"Modèle quantifié sauvegardé dans {path}")


def quantize_model(float_model, X_calib):
    """Pipeline complet : calibration + quantification."""
    ranges = collect_activation_ranges(float_model, X_calib)
    qmodel = QuantizedLeNet(float_model, ranges)
    return qmodel


if __name__ == '__main__':
    # Test rapide sur le modèle (poids aléatoires si pas de vrai .npz)
    from model import LeNet

    np.random.seed(0)
    model = LeNet()
    try:
        model.load('lenet_mnist.npz')
    except FileNotFoundError:
        print("(pas de .npz trouvé, test avec poids aléatoires)")

    # Données de calibration synthétiques
    X_calib = np.random.rand(100, 28, 28, 1).astype(np.float32)

    qmodel = quantize_model(model, X_calib)

    # Test du forward quantifié
    X_test = np.random.rand(8, 28, 28, 1).astype(np.float32)
    logits_float = model.forward(X_test)
    logits_quant = qmodel.forward(X_test)

    print("\nComparaison logits float vs quantifié (8 échantillons):")
    pred_f = np.argmax(logits_float, axis=1)
    pred_q = np.argmax(logits_quant, axis=1)
    print(f"  prédictions float    : {pred_f}")
    print(f"  prédictions quantifié: {pred_q}")
    print(f"  accord: {(pred_f == pred_q).sum()}/8")