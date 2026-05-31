"""
Évaluation de la fidélité du modèle quantifié.

Économiser de la mémoire ne vaut rien si la précision s'effondre.
Ce script mesure la 5e dimension, la plus importante :
    -> combien de précision perd-on en quantifiant ?

Compare sur le jeu de test MNIST:
    - accuracy du modèle float32
    - accuracy du modèle int8 quantifié
    - taux d'accord entre les deux (sur quelles images divergent-ils ?)
"""

import numpy as np

from model import LeNet
from mnist_loader import load_mnist
from quantized_model import quantize_model


def evaluate_accuracy(predict_fn, X, Y, batch_size=128):
    """Accuracy d'un modèle sur un dataset."""
    correct = 0
    for i in range(0, len(X), batch_size):
        preds = predict_fn(X[i:i+batch_size])
        correct += (preds == Y[i:i+batch_size]).sum()
    return correct / len(X)


def main(model_path='lenet_mnist.npz', n_calib=200, n_test=2000):
    # Modèle float
    model = LeNet()
    model.load(model_path)

    # Données : un sous-ensemble pour la calibration, un autre pour le test
    _, _, X_test, Y_test = load_mnist(max_test=n_calib + n_test)
    X_calib = X_test[:n_calib]
    X_eval = X_test[n_calib:n_calib + n_test]
    Y_eval = Y_test[n_calib:n_calib + n_test]

    # Quantification
    qmodel = quantize_model(model, X_calib)

    # Évaluations
    print(f"\nÉvaluation sur {len(X_eval)} images de test...")
    acc_float = evaluate_accuracy(lambda x: model.predict(x), X_eval, Y_eval)
    acc_quant = evaluate_accuracy(lambda x: qmodel.predict(x), X_eval, Y_eval)

    # Taux d'accord
    pred_f = model.predict(X_eval)
    pred_q = qmodel.predict(X_eval)
    agreement = (pred_f == pred_q).mean()

    print("\n" + "=" * 50)
    print("FIDÉLITÉ DU MODÈLE QUANTIFIÉ")
    print("=" * 50)
    print(f"  Accuracy float32  : {acc_float*100:.2f} %")
    print(f"  Accuracy int8     : {acc_quant*100:.2f} %")
    print(f"  Perte de précision: {(acc_float-acc_quant)*100:+.2f} points")
    print(f"  Accord float/int8 : {agreement*100:.2f} %")
    print("=" * 50)

    if acc_float - acc_quant > 0.02:
        print("\n[!] Perte > 2 points. Pistes d'amélioration :")
        print("    - quantification per-channel des poids de conv")
        print("    - calibration sur plus d'échantillons")
        print("    - quantization-aware training (QAT)")
    else:
        print("\n[OK] Perte de précision acceptable (< 2 points).")


if __name__ == '__main__':
    main()  