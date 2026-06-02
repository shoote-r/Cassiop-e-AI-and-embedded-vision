# Cassiopée n°79 — IA et Vision Embarquée

Classifieur de chiffres manuscrits (MNIST) implémenté **from scratch en NumPy**,
optimisé pour la vision embarquée via **quantification full-integer**.
Inférence temps réel via webcam.

## Prérequis

- Python 3.10 ou supérieur
- Une webcam (pour l'inférence temps réel)

## Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>
pip install -r requirements.txt
```

## Utilisation

Deux cas d'usage selon votre objectif.

### Cas 1 — Utilisation standard

Entraîner le modèle puis l'utiliser en inférence webcam.

```bash
python train.py            # entraîne le modèle (~4 min)
python inference_cam.py    # lance l'inférence webcam temps réel
```

Présentez un chiffre écrit au stylo noir sur papier blanc dans le carré vert
affiché à l'écran. Appuyez sur `q` pour quitter.

### Cas 2 — Évaluation de la quantification

Nécessite d'avoir déjà entraîné le modèle (Cas 1) ou d'utiliser le fichier
`lenet_mnist.npz` fourni.

```bash
python quantization.py        # vérification des primitives
python eval_quantization.py   # mesure de fidélité float vs int8
python benchmark.py           # mesure des économies (taille, RAM, temps, énergie)
```

## Structure du projet

| Fichier                | Rôle                                                       |
|------------------------|------------------------------------------------------------|
| `mnist_loader.py`      | Téléchargement et chargement du dataset MNIST              |
| `layers.py`            | Briques du réseau (Conv2D, MaxPool, Dense, ReLU, Softmax)  |
| `model.py`             | Assemblage du LeNet, sauvegarde/chargement des poids       |
| `train.py`             | Boucle d'entraînement et augmentation de données           |
| `gradient_check.py`    | Validation de la rétropropagation (différences finies)     |
| `preprocessing.py`     | Pipeline de transformation image webcam → format MNIST     |
| `inference_cam.py`     | Boucle webcam OpenCV et inférence temps réel               |
| `quantization.py`      | Primitives de quantification (QTensor, scales)             |
| `quantized_model.py`   | Modèle quantifié full-integer (calibration + forward int8) |
| `benchmark.py`         | Mesures comparatives float vs int8                         |
| `eval_quantization.py` | Évaluation de la fidélité du modèle quantifié              |
| `lenet_mnist.npz`      | Poids du modèle entraîné (float32)                         |
| `lenet_quant.npz`      | Poids du modèle quantifié (int8)                           |

- NB : architecture inspirée du LeNet après avoir essayé un MLP sans succès. La quantification a été réalisée sur les poids et les fonctions d'activation.

## Auteurs

Projet réalisé dans le cadre du projet Cassiopée n°79 à Télécom SudParis,
encadré par M. Ghalid Abib.