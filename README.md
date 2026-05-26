Projet Cassiopée...
Branche conversion_Optimisation ayant pour but d'introduire le pruning et la quantification 

# Fonctionnement 
Voici la structure utilisée : 

cnn_mnist/
├── mnist_loader.py      Chargement du dataset MNIST
├── layers.py            Briques du réseau (conv, pool, dense, activations)
├── model.py             Assemblage LeNet
├── gradient_check.py    Validation de la backprop
├── train.py             Entraînement
├── preprocessing.py     Prétraitement image webcam → MNIST
├── inference_cam.py     Inférence temps réel webcam
│
├── mnist_data/          Dataset téléchargé (4 fichiers .gz)
└── lenet_mnist.npz      Poids entraînés


NB : gradient.py a pour unique but de vérifier que la backprop est correcte, sinon le fichier n'a pas d'impact sur le fonctionnement du réseau.

