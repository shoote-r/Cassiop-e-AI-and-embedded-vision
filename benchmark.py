"""
Benchmark : mesure l'impact de la quantification.

Quatre dimensions mesurées:
    1. TAILLE DISQUE  : octets du fichier .npz (mesure directe, fiable)
    2. RAM            : empreinte des poids + plus gros tampon d'activations
                        (mesure analytique, reproductible)
    3. TEMPS          : durée moyenne d'une inférence (mesure empirique)
    4. ÉNERGIE        : PROXY uniquement — nombre de MACs et octets déplacés.
                        L'énergie réelle se mesure avec un wattmètre, pas en
                        Python. On fournit un indicateur indirect.

Avertissement honnête sur le TEMPS:
    NumPy est optimisé pour le float32 (BLAS). Une inférence "int8" simulée
    en NumPy n'est PAS forcément plus rapide — elle peut même être plus lente,
    car on ajoute des étapes de (dé)quantification et l'int8 ne bénéficie pas
    du même chemin optimisé. Le gain de vitesse de l'entier est réel sur
    microcontrôleur (pas de FPU) ou avec du SIMD entier, pas ici.
    On mesure quand même, mais on interprète avec prudence.
"""

import os
import time
import numpy as np

from layers import Conv2D, Dense, ReLU, MaxPool2D, Flatten


# ============================================================================
# 1. Taille disque
# ============================================================================

def measure_disk_size(path):
    """Taille d'un fichier en octets. Mesure directe."""
    return os.path.getsize(path)


# ============================================================================
# 2. RAM : empreinte analytique
# ============================================================================

def measure_param_memory_float(model):
    """Octets occupés par les poids du modèle float (float32 = 4 octets)."""
    total = 0
    for layer in model.layers:
        for name, p, g in layer.params():
            total += p.size * 4  # float32
    return total


def measure_param_memory_quantized(qmodel):
    """Octets occupés par les poids du modèle quantifié (int8 = 1 octet)."""
    total = 0
    for name, qt in qmodel.qweights.items():
        total += qt.q.size * 1          # poids int8
    for name, qb in qmodel.qbias.items():
        total += qb.size * 4            # biais int32
    # Les scales/zero_points sont négligeables mais on les compte
    total += len(qmodel.act_scales) * 4
    total += len(qmodel.act_zps) * 4
    return total


def measure_activation_memory(model, input_shape, dtype_bytes):
    """
    Plus gros tampon d'activation nécessaire pendant un forward.

    En inférence, on n'a besoin que de l'activation courante (la précédente
    peut être libérée). Le pic de RAM d'activation = la plus grande sortie
    de couche.

    Args:
        dtype_bytes: 4 pour float32, 1 pour int8
    """
    x = np.zeros(input_shape, dtype=np.float32)
    peak = x.size

    for layer in model.layers:
        x = layer.forward(x)
        peak = max(peak, x.size)

    return peak * dtype_bytes


# ============================================================================
# 3. Temps d'inférence
# ============================================================================

def measure_inference_time(predict_fn, X, n_runs=5, warmup=2):
    """
    Mesure le temps moyen d'inférence.

    Args:
        predict_fn: fonction qui prend X et fait une inférence
        X: batch d'entrée
        n_runs: nombre de mesures
        warmup: itérations de chauffe non comptées (caches, etc.)

    Returns:
        dict avec temps moyen, écart-type, temps par image
    """
    # Chauffe
    for _ in range(warmup):
        predict_fn(X)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        'mean_s': float(times.mean()),
        'std_s': float(times.std()),
        'per_image_ms': float(times.mean() / len(X) * 1000),
    }


# ============================================================================
# 4. Proxy énergie : compte de MACs et d'octets déplacés
# ============================================================================

def count_macs(model, input_shape):
    """
    Compte les MACs (Multiply-ACcumulate) d'un forward.

    Le MAC est l'opération de base d'un réseau (un produit + une addition).
    C'est une métrique matériel-indépendante du coût de calcul.

    Note: le NOMBRE de MACs ne change PAS avec la quantification (même
    architecture). Ce qui change, c'est le COÛT unitaire d'un MAC : un
    MAC int8 consomme bien moins qu'un MAC float32 sur l'embarqué.
    """
    macs = 0
    x = np.zeros(input_shape, dtype=np.float32)

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            out = layer.forward(x)
            # Chaque pixel de sortie = (kh * kw * C_in) MACs
            _, H_out, W_out, C_out = out.shape
            k = layer.kernel_size
            macs_layer = H_out * W_out * C_out * (k * k * layer.in_channels)
            macs += macs_layer
            x = out
        elif isinstance(layer, Dense):
            out = layer.forward(x)
            # in_features * out_features MACs
            macs += layer.in_features * layer.out_features
            x = out
        else:
            x = layer.forward(x)

    return macs


# Coûts énergétiques RELATIFS approximatifs (ordre de grandeur, littérature
# type Horowitz "Computing's Energy Problem"). À 45nm, en picojoules :
#   - addition int8   : ~0.03 pJ      - addition float32 : ~0.9 pJ
#   - mult int8       : ~0.2 pJ       - mult float32     : ~3.7 pJ
# On retient un ratio simple : un MAC int8 coûte ~1/15 d'un MAC float32.
ENERGY_PER_MAC_FLOAT32 = 4.6   # pJ (mult + add)
ENERGY_PER_MAC_INT8 = 0.23     # pJ (mult + add)


def estimate_energy_proxy(macs, mode='float32'):
    """
    PROXY d'énergie de calcul, en picojoules.

    ATTENTION : ceci est une ESTIMATION INDIRECTE basée sur des coûts
    unitaires de la littérature. L'énergie réelle dépend du matériel,
    des accès mémoire, du compilateur, etc. À mesurer physiquement pour
    des chiffres fiables. Utile ici uniquement pour comparer float vs int8
    en ordre de grandeur.
    """
    cost = ENERGY_PER_MAC_FLOAT32 if mode == 'float32' else ENERGY_PER_MAC_INT8
    return macs * cost


# ============================================================================
# Rapport comparatif complet
# ============================================================================

def full_report(float_model, qmodel, X_test,
                 float_npz='lenet_mnist.npz', quant_npz='lenet_quant.npz'):
    """
    Produit un rapport comparatif float vs quantifié sur les 4 dimensions.
    """
    input_shape = (1,) + X_test.shape[1:]

    print("=" * 60)
    print("RAPPORT COMPARATIF : float32 vs int8 quantifié")
    print("=" * 60)

    # --- 1. Taille disque ---
    print("\n1. TAILLE DISQUE")
    size_f = measure_disk_size(float_npz) if os.path.exists(float_npz) else None
    size_q = measure_disk_size(quant_npz) if os.path.exists(quant_npz) else None
    if size_f and size_q:
        print(f"   float32  : {size_f:>10,} octets ({size_f/1024:.1f} Ko)")
        print(f"   int8     : {size_q:>10,} octets ({size_q/1024:.1f} Ko)")
        print(f"   gain     : {(1 - size_q/size_f)*100:.1f} % "
              f"(facteur {size_f/size_q:.2f}x)")
    else:
        print("   (fichiers .npz absents — sauvegardez les deux modèles d'abord)")

    # --- 2. RAM ---
    print("\n2. RAM (empreinte analytique)")
    pmem_f = measure_param_memory_float(float_model)
    pmem_q = measure_param_memory_quantized(qmodel)
    act_f = measure_activation_memory(float_model, input_shape, dtype_bytes=4)
    act_q = measure_activation_memory(float_model, input_shape, dtype_bytes=1)
    print(f"   Poids    float32 : {pmem_f:>9,} o   int8 : {pmem_q:>9,} o   "
          f"({(1-pmem_q/pmem_f)*100:.1f} %)")
    print(f"   Activ.   float32 : {act_f:>9,} o   int8 : {act_q:>9,} o   "
          f"({(1-act_q/act_f)*100:.1f} %)")
    total_f = pmem_f + act_f
    total_q = pmem_q + act_q
    print(f"   TOTAL    float32 : {total_f:>9,} o   int8 : {total_q:>9,} o   "
          f"({(1-total_q/total_f)*100:.1f} %)")

    # --- 3. Temps ---
    print("\n3. TEMPS D'INFÉRENCE")
    t_float = measure_inference_time(lambda x: float_model.predict(x), X_test)
    t_quant = measure_inference_time(lambda x: qmodel.predict(x), X_test)
    print(f"   float32  : {t_float['per_image_ms']:.3f} ms/image")
    print(f"   int8     : {t_quant['per_image_ms']:.3f} ms/image")
    ratio = t_quant['per_image_ms'] / t_float['per_image_ms']
    if ratio > 1:
        print(f"   --> l'int8 simulé est {ratio:.2f}x PLUS LENT en NumPy "
              f"(normal, cf. avertissement)")
    else:
        print(f"   --> gain {(1-ratio)*100:.1f} %")

    # --- 4. Proxy énergie ---
    print("\n4. ÉNERGIE (PROXY — estimation indirecte, pas une mesure)")
    macs = count_macs(float_model, input_shape)
    e_float = estimate_energy_proxy(macs, 'float32')
    e_quant = estimate_energy_proxy(macs, 'int8')
    print(f"   MACs par inférence : {macs:,}")
    print(f"   énergie proxy float32 : {e_float/1e6:.2f} µJ")
    print(f"   énergie proxy int8    : {e_quant/1e6:.2f} µJ")
    print(f"   gain proxy : {(1-e_quant/e_float)*100:.1f} % "
          f"(facteur {e_float/e_quant:.1f}x)")
    print("   [!] proxy basé sur coûts unitaires littérature — "
          "à confirmer au wattmètre")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    from model import LeNet
    from quantized_model import quantize_model

    np.random.seed(0)
    model = LeNet()
    try:
        model.load('lenet_mnist.npz')
    except FileNotFoundError:
        print("(pas de .npz — test avec poids aléatoires)")

    X_calib = np.random.rand(100, 28, 28, 1).astype(np.float32)
    X_test = np.random.rand(32, 28, 28, 1).astype(np.float32)

    qmodel = quantize_model(model, X_calib)
    qmodel.save('lenet_quant.npz')

    print()
    full_report(model, qmodel, X_test)