import numpy as np

# --- 1. INITIALISATION DES POIDS (Les matrices brutes) ---
def init_mobilenet_weights():
    weights = {}
    # L'astuce : multiplier par 0.1 pour avoir de petites valeurs aléatoires
    
    # Couche 1 : Convolution standard (Entrée: 96x96x1 -> Sortie: 48x48x8)
    # Filtres: 3x3, 1 canal d'entrée, 8 canaux de sortie
    weights['W_conv1'] = np.random.randn(3, 3, 1, 8) * 0.1
    weights['b_conv1'] = np.zeros(8)
    
    # Bloc Séparable 1 (Passage de 8 à 16 canaux)
    # Depthwise: 1 filtre 3x3 par canal d'entrée (donc 8 filtres)
    weights['W_dw1'] = np.random.randn(3, 3, 8) * 0.1
    weights['b_dw1'] = np.zeros(8)
    # Pointwise: Filtres 1x1, regarde 8 canaux, crée 16 canaux
    weights['W_pw1'] = np.random.randn(1, 1, 8, 16) * 0.1
    weights['b_pw1'] = np.zeros(16)
    
    # Bloc Séparable 2 (Passage de 16 à 32 canaux)
    weights['W_dw2'] = np.random.randn(3, 3, 16) * 0.1
    weights['b_dw2'] = np.zeros(16)
    weights['W_pw2'] = np.random.randn(1, 1, 16, 32) * 0.1
    weights['b_pw2'] = np.zeros(32)
    
    # Couche Dense de Sortie (32 entrées -> 5 classes pour tes emojis)
    weights['W_dense'] = np.random.randn(32, 5) * 0.1
    weights['b_dense'] = np.zeros(5)
    
    return weights

# --- 2. LES OPÉRATIONS MATHÉMATIQUES (Forward Pass) ---
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def standard_conv2d(image, W, b, stride=2):
    """Convolution classique codée à la main"""
    h_in, w_in, c_in = image.shape
    f_h, f_w, _, c_out = W.shape
    h_out, w_out = (h_in - f_h) // stride + 1, (w_in - f_w) // stride + 1
    
    out = np.zeros((h_out, w_out, c_out))
    for c in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                region = image[i*stride : i*stride+f_h, j*stride : j*stride+f_w, :]
                out[i, j, c] = np.sum(region * W[:, :, :, c]) + b[c]
    return out

def depthwise_conv2d(image, W, b, stride=1):
    """Filtre spatial : 1 filtre par canal (pas de mélange)"""
    h_in, w_in, c_in = image.shape
    f_h, f_w, _ = W.shape
    h_out, w_out = (h_in - f_h) // stride + 1, (w_in - f_w) // stride + 1
    
    out = np.zeros((h_out, w_out, c_in))
    for c in range(c_in): # Note: on ne boucle que sur c_in
        for i in range(h_out):
            for j in range(w_out):
                region = image[i*stride : i*stride+f_h, j*stride : j*stride+f_w, c]
                out[i, j, c] = np.sum(region * W[:, :, c]) + b[c]
    return out

def pointwise_conv2d(image, W, b):
    """Filtre 1x1 : mélange des canaux"""
    h_in, w_in, c_in = image.shape
    _, _, _, c_out = W.shape
    
    out = np.zeros((h_in, w_in, c_out))
    for i in range(h_in):
        for j in range(w_in):
            pixel = image[i, j, :] # Le pixel sur toute sa profondeur
            for c in range(c_out):
                out[i, j, c] = np.sum(pixel * W[0, 0, :, c]) + b[c]
    return out

def global_average_pooling(image):
    """Réduit l'image 2D en un simple vecteur 1D en faisant la moyenne"""
    # Moyenne sur les axes spatiaux (0 et 1), on garde les canaux (axe 2)
    return np.mean(image, axis=(0, 1))

# --- 3. L'ASSEMBLAGE (Le passage complet de l'image) ---
def forward_mobilenet(image, weights):
    # Image d'entrée : 96x96x1
    
    # 1. Conv Initiale
    x = standard_conv2d(image, weights['W_conv1'], weights['b_conv1'], stride=2)
    x = relu(x)
    
    # 2. Bloc Séparable 1
    x = depthwise_conv2d(x, weights['W_dw1'], weights['b_dw1'], stride=2)
    x = relu(x)
    x = pointwise_conv2d(x, weights['W_pw1'], weights['b_pw1'])
    x = relu(x)
    
    # 3. Bloc Séparable 2
    x = depthwise_conv2d(x, weights['W_dw2'], weights['b_dw2'], stride=2)
    x = relu(x)
    x = pointwise_conv2d(x, weights['W_pw2'], weights['b_pw2'])
    x = relu(x)
    
    # 4. Sortie
    x = global_average_pooling(x) # Devient un vecteur de taille 32
    z = np.dot(x, weights['W_dense']) + weights['b_dense']
    
    return softmax(z)

# --- TEST ---
if __name__ == "__main__":
    # Simuler une image 96x96 en niveaux de gris (valeurs entre 0 et 1)
    image_test = np.random.rand(96, 96, 1)
    mes_poids = init_mobilenet_weights()
    
    prediction = forward_mobilenet(image_test, mes_poids)
    print("Probabilités pour les 5 classes :", prediction)