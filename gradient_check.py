"""
Gradient checking en float64 pour avoir une précision numérique suffisante.

Le truc : NumPy en float32 n'a que ~7 chiffres significatifs.
Pour une différence finie centrée, on a:
    grad_num = (loss(p + eps) - loss(p - eps)) / (2 * eps)
La soustraction au numérateur peut perdre beaucoup de chiffres si les deux
losses sont proches. En float32, sur un réseau de 10 couches, l'accumulation
des erreurs d'arrondi rend les gradients numériques inexploitables.

Solution : tout convertir en float64 LE TEMPS DU TEST.
"""

import numpy as np
from model import LeNet


def cast_model_to_float64(model):
    """Convertit tous les poids et leurs gradients en float64 (in-place)."""
    for layer in model.layers:
        for attr in ['W', 'b', 'dW', 'db']:
            if hasattr(layer, attr):
                setattr(layer, attr, getattr(layer, attr).astype(np.float64))


def numerical_gradient_for_param(model, x, y, param, n_samples=10, epsilon=1e-5):
    n_samples = min(n_samples, param.size)
    flat_indices = np.random.choice(param.size, n_samples, replace=False)

    grad_nums = []
    for flat_idx in flat_indices:
        idx = np.unravel_index(flat_idx, param.shape)
        original = param[idx]

        param[idx] = original + epsilon
        loss_plus, _ = model.criterion.forward(model.forward(x), y)

        param[idx] = original - epsilon
        loss_minus, _ = model.criterion.forward(model.forward(x), y)

        param[idx] = original
        grad_nums.append((loss_plus - loss_minus) / (2 * epsilon))

    return flat_indices, grad_nums


def relative_error(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-12)


def main():
    np.random.seed(42)
    model = LeNet()
    cast_model_to_float64(model)

    x = (np.random.rand(2, 28, 28, 1) * 0.5).astype(np.float64)
    y = np.array([3, 7], dtype=np.int32)

    model.loss_and_backward(x, y)

    print("Gradient checking en float64 :")
    print("(< 1e-7 parfait, < 1e-5 OK, > 1e-3 bug)\n")

    all_ok = True
    for layer in model.layers:
        for name, p, g in layer.params():
            indices, grad_nums = numerical_gradient_for_param(model, x, y, p)
            errors = []
            for flat_idx, g_num in zip(indices, grad_nums):
                idx = np.unravel_index(flat_idx, p.shape)
                errors.append(relative_error(g[idx], g_num))

            max_err = max(errors)
            mean_err = sum(errors) / len(errors)
            status = "OK" if max_err < 1e-5 else "BUG"
            if max_err >= 1e-5:
                all_ok = False
            print(f"  [{status}] {layer.name}.{name}: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

    print("\n" + ("Tous les gradients sont corrects." if all_ok else "Bug detecte."))


if __name__ == '__main__':
    main()