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



## Quantification 
1. **$python quantization.py** -> met en place la quantification i.e vérifie les primitives (comment quantifier une valeur)
2. **$python eval_quantization.py** -> évalue la perte de précision causée par la quantification 
3. **$python benchmark.py** -> impact quantification sur espace disque, RAM et temps (attention l'évalualtion sur le temps est ric rac)

- **quantized_model.py** -> applique la quantification au réseau entier

## Pruning 
Le pruning s'effectue par étapes : entraînement -> pruning -> reentraînememnt -> pruning -> ...
On a 2 types de pruning : 
1. pruning non-structuré  : on met à zéro les poids individuels les plus petits, cela dit on obtient un gain de mémoire uniquement si on stocke ensuite les poids dans des matrices creuses et que le matériel embarqué sait les exploiter 
2. pruning structuré : ici on supprime des structures entières i.e un filtre de convolution ou une couche dense donc on réduit le modèle. Cela impact beaucoup la précision et il faut donc effectuer plus de fine tuning.