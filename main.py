import os
import csv
import time
import numpy as np
from PIL import Image

def augment_image_numpy(image_array):
    """
    Applique des transformations aléatoires légères à une image NumPy (H, W, C).
    """
    aug_image = image_array.copy()

    # 1. Flip horizontal aléatoire (50% de chance)
    if np.random.rand() > 0.5:
        # np.fliplr inverse l'image de gauche à droite
        aug_image = np.fliplr(aug_image)

    # 2. Ajout de bruit gaussien léger
    if np.random.rand() > 0.5:
        # Ajoute un bruit aléatoire d'une variance de 0.02
        noise = np.random.randn(*aug_image.shape) * 0.02
        aug_image = aug_image + noise

    # 3. Translation légère (décalage de quelques pixels)
    if np.random.rand() > 0.5:
        shift_x = np.random.randint(-3, 4) # Décalage entre -3 et +3 pixels
        shift_y = np.random.randint(-3, 4)
        # On utilise roll (attention, ça crée un effet de boucle sur les bords, 
        # mais c'est suffisant pour un petit modèle maison)
        aug_image = np.roll(aug_image, shift=shift_y, axis=0)
        aug_image = np.roll(aug_image, shift=shift_x, axis=1)

    # On s'assure que les valeurs restent bien entre 0 et 1 (à cause du bruit)
    aug_image = np.clip(aug_image, 0.0, 1.0)
    
    return aug_image
# -----------------------
# MobileNetV1 from scratch
# Forward + backward (pedagogical, not optimized)
# -----------------------

# --- Activations and losses ---

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, cache_x):
    return dout * (cache_x > 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def softmax_cross_entropy_backward(predictions, target_class_idx):
    dz = predictions.copy()
    dz[target_class_idx] -= 1.0
    return dz

# --- Layer backward helpers ---

def dense_backward(dout, cache_x, W):
    # cache_x: vector (C,), dout: vector (K,), W: (C, K)
    dW = np.outer(cache_x, dout)
    db = dout.copy()
    d_x = np.dot(W, dout)
    return d_x, dW, db

def global_average_pooling(image):
    return np.mean(image, axis=(0, 1))

def global_average_pooling_backward(dout, shape_before_pool):
    h, w, c = shape_before_pool
    return np.ones((h, w, c)) * (dout / (h * w))

# --- Convolution (naïf) ---

def standard_conv2d(image, W, b, stride=2):
    h_in, w_in, c_in = image.shape
    f_h, f_w, _, c_out = W.shape
    h_out = (h_in - f_h) // stride + 1
    w_out = (w_in - f_w) // stride + 1
    out = np.zeros((h_out, w_out, c_out))
    for c in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                region = image[i*stride:i*stride+f_h, j*stride:j*stride+f_w, :]
                out[i, j, c] = np.sum(region * W[:, :, :, c]) + b[c]
    return out

def standard_conv2d_backward(dout, cache_image_in, W, stride=2):
    h_in, w_in, c_in = cache_image_in.shape
    f_h, f_w, _, c_out = W.shape
    h_out, w_out, _ = dout.shape
    dW = np.zeros_like(W)
    db = np.sum(dout, axis=(0, 1))
    d_image_in = np.zeros_like(cache_image_in)
    for c in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                region = cache_image_in[h_start:h_start+f_h, w_start:w_start+f_w, :]
                dW[:, :, :, c] += region * dout[i, j, c]
                d_image_in[h_start:h_start+f_h, w_start:w_start+f_w, :] += W[:, :, :, c] * dout[i, j, c]
    return d_image_in, dW, db

# --- Depthwise ---

def depthwise_conv2d(image, W, b, stride=1):
    h_in, w_in, c_in = image.shape
    f_h, f_w, _ = W.shape
    h_out = (h_in - f_h) // stride + 1
    w_out = (w_in - f_w) // stride + 1
    out = np.zeros((h_out, w_out, c_in))
    for c in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                region = image[i*stride:i*stride+f_h, j*stride:j*stride+f_w, c]
                out[i, j, c] = np.sum(region * W[:, :, c]) + b[c]
    return out

def depthwise_conv2d_backward(dout, cache_image_in, W, stride=1):
    h_in, w_in, c_in = cache_image_in.shape
    f_h, f_w, _ = W.shape
    h_out, w_out, _ = dout.shape
    dW = np.zeros_like(W)
    db = np.sum(dout, axis=(0, 1))
    d_image_in = np.zeros_like(cache_image_in)
    for c in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                region = cache_image_in[h_start:h_start+f_h, w_start:w_start+f_w, c]
                dW[:, :, c] += region * dout[i, j, c]
                d_image_in[h_start:h_start+f_h, w_start:w_start+f_w, c] += W[:, :, c] * dout[i, j, c]
    return d_image_in, dW, db

# --- Pointwise ---

def pointwise_conv2d(image, W, b):
    h_in, w_in, c_in = image.shape
    _, _, _, c_out = W.shape
    out = np.zeros((h_in, w_in, c_out))
    for i in range(h_in):
        for j in range(w_in):
            pixel = image[i, j, :]
            for c in range(c_out):
                out[i, j, c] = np.sum(pixel * W[0, 0, :, c]) + b[c]
    return out

def pointwise_conv2d_backward(dout, cache_image_in, W):
    h_in, w_in, c_in = cache_image_in.shape
    _, _, _, c_out = W.shape
    dW = np.zeros_like(W)
    db = np.sum(dout, axis=(0, 1))
    d_image_in = np.zeros_like(cache_image_in)
    for i in range(h_in):
        for j in range(w_in):
            for c in range(c_out):
                dW[0, 0, :, c] += cache_image_in[i, j, :] * dout[i, j, c]
                d_image_in[i, j, :] += W[0, 0, :, c] * dout[i, j, c]
    return d_image_in, dW, db

# --- Weights init ---

def init_mobilenet_weights():
    weights = {}
    weights['W_conv1'] = np.random.randn(3, 3, 1, 8) * np.sqrt(2.0 / (3 * 3 * 1))
    weights['b_conv1'] = np.zeros(8)
    weights['W_dw1'] = np.random.randn(3, 3, 8) * np.sqrt(2.0 / (3 * 3))
    weights['b_dw1'] = np.zeros(8)
    weights['W_pw1'] = np.random.randn(1, 1, 8, 16) * np.sqrt(2.0 / 8)
    weights['b_pw1'] = np.zeros(16)
    weights['W_dw2'] = np.random.randn(3, 3, 16) * np.sqrt(2.0 / (3 * 3))
    weights['b_dw2'] = np.zeros(16)
    weights['W_pw2'] = np.random.randn(1, 1, 16, 32) * np.sqrt(2.0 / 16)
    weights['b_pw2'] = np.zeros(32)
    weights['W_dense'] = np.random.randn(32, 5) * np.sqrt(2.0 / 32)
    weights['b_dense'] = np.zeros(5)
    return weights

# --- Forward & train step ---

def forward_mobilenet(image, weights):
    x = standard_conv2d(image, weights['W_conv1'], weights['b_conv1'], stride=2)
    z_conv1 = x.copy()
    a_conv1 = relu(z_conv1)

    z_dw1 = depthwise_conv2d(a_conv1, weights['W_dw1'], weights['b_dw1'], stride=2)
    a_dw1 = relu(z_dw1)
    z_pw1 = pointwise_conv2d(a_dw1, weights['W_pw1'], weights['b_pw1'])
    a_pw1 = relu(z_pw1)

    z_dw2 = depthwise_conv2d(a_pw1, weights['W_dw2'], weights['b_dw2'], stride=2)
    a_dw2 = relu(z_dw2)
    z_pw2 = pointwise_conv2d(a_dw2, weights['W_pw2'], weights['b_pw2'])
    a_pw2 = relu(z_pw2)

    pooled = global_average_pooling(a_pw2)
    z = np.dot(pooled, weights['W_dense']) + weights['b_dense']
    return z, {
        'z_conv1': z_conv1,
        'a_conv1': a_conv1,
        'z_dw1': z_dw1,
        'a_dw1': a_dw1,
        'z_pw1': z_pw1,
        'a_pw1': a_pw1,
        'z_dw2': z_dw2,
        'a_dw2': a_dw2,
        'z_pw2': z_pw2,
        'a_pw2': a_pw2,
        'pooled_shape': a_pw2.shape
    }

def train_step(image, target_class, weights, learning_rate=0.01):
    # forward
    z, cache = forward_mobilenet(image, weights)
    predictions = softmax(z)

    # loss grad
    dZ_dense = softmax_cross_entropy_backward(predictions, target_class)

    # dense backward
    # dense backward
    dA_pool, dW_dense, db_dense = dense_backward(dZ_dense, global_average_pooling(cache['a_pw2']), weights['W_dense'])

    # pooling backward
    dA_pw2 = global_average_pooling_backward(dA_pool, cache['pooled_shape'])

    # block2 backward
    dZ_pw2 = relu_backward(dA_pw2, cache['z_pw2'])
    dA_dw2, dW_pw2, db_pw2 = pointwise_conv2d_backward(dZ_pw2, cache['a_dw2'], weights['W_pw2'])
    dZ_dw2 = relu_backward(dA_dw2, cache['z_dw2'])
    dA_pw1, dW_dw2, db_dw2 = depthwise_conv2d_backward(dZ_dw2, cache['a_pw1'], weights['W_dw2'], stride=2)

    # block1 backward
    dZ_pw1 = relu_backward(dA_pw1, cache['z_pw1'])
    dA_dw1, dW_pw1, db_pw1 = pointwise_conv2d_backward(dZ_pw1, cache['a_dw1'], weights['W_pw1'])
    dZ_dw1 = relu_backward(dA_dw1, cache['z_dw1'])
    dA_conv1, dW_dw1, db_dw1 = depthwise_conv2d_backward(dZ_dw1, cache['a_conv1'], weights['W_dw1'], stride=2)

    # initial conv backward
    dZ_conv1 = relu_backward(dA_conv1, cache['z_conv1'])
    dImage, dW_conv1, db_conv1 = standard_conv2d_backward(dZ_conv1, image, weights['W_conv1'], stride=2)

    # update weights
    weights['W_dense'] -= learning_rate * dW_dense
    weights['b_dense'] -= learning_rate * db_dense

    weights['W_pw2'] -= learning_rate * dW_pw2
    weights['b_pw2'] -= learning_rate * db_pw2
    weights['W_dw2'] -= learning_rate * dW_dw2
    weights['b_dw2'] -= learning_rate * db_dw2

    weights['W_pw1'] -= learning_rate * dW_pw1
    weights['b_pw1'] -= learning_rate * db_pw1
    weights['W_dw1'] -= learning_rate * dW_dw1
    weights['b_dw1'] -= learning_rate * db_dw1

    weights['W_conv1'] -= learning_rate * dW_conv1
    weights['b_conv1'] -= learning_rate * db_conv1

    loss = -np.log(predictions[target_class] + 1e-12)
    return loss, predictions, weights

def compute_gradients(image, target_class, weights):
    # 1. Forward pass (identique à avant)
    z, cache = forward_mobilenet(image, weights)
    predictions = softmax(z)
    loss = -np.log(predictions[target_class] + 1e-12)

    # 2. Backward pass (identique à avant)
    dZ_dense = softmax_cross_entropy_backward(predictions, target_class)
    dA_pool, dW_dense, db_dense = dense_backward(dZ_dense, global_average_pooling(cache['a_pw2']), weights['W_dense'])
    
    dA_pw2 = global_average_pooling_backward(dA_pool, cache['pooled_shape'])
    
    dZ_pw2 = relu_backward(dA_pw2, cache['z_pw2'])
    dA_dw2, dW_pw2, db_pw2 = pointwise_conv2d_backward(dZ_pw2, cache['a_dw2'], weights['W_pw2'])
    dZ_dw2 = relu_backward(dA_dw2, cache['z_dw2'])
    dA_pw1, dW_dw2, db_dw2 = depthwise_conv2d_backward(dZ_dw2, cache['a_pw1'], weights['W_dw2'], stride=2)

    dZ_pw1 = relu_backward(dA_pw1, cache['z_pw1'])
    dA_dw1, dW_pw1, db_pw1 = pointwise_conv2d_backward(dZ_pw1, cache['a_dw1'], weights['W_pw1'])
    dZ_dw1 = relu_backward(dA_dw1, cache['z_dw1'])
    dA_conv1, dW_dw1, db_dw1 = depthwise_conv2d_backward(dZ_dw1, cache['a_conv1'], weights['W_dw1'], stride=2)

    dZ_conv1 = relu_backward(dA_conv1, cache['z_conv1'])
    dImage, dW_conv1, db_conv1 = standard_conv2d_backward(dZ_conv1, image, weights['W_conv1'], stride=2)

    # 3. ON NE MET PLUS À JOUR ICI. On stocke les gradients dans un dictionnaire.
    grads = {
        'W_dense': dW_dense, 'b_dense': db_dense,
        'W_pw2': dW_pw2, 'b_pw2': db_pw2,
        'W_dw2': dW_dw2, 'b_dw2': db_dw2,
        'W_pw1': dW_pw1, 'b_pw1': db_pw1,
        'W_dw1': dW_dw1, 'b_dw1': db_dw1,
        'W_conv1': dW_conv1, 'b_conv1': db_conv1
    }
    
    return loss, predictions, grads
# --- Training loop ---

def train_mobilenet_from_scratch(X_train, Y_train, epochs=5, learning_rate=0.005, batch_size=16, max_samples=None):
    # Initialisation de He (plus stable que randn * 0.1)
    weights = init_mobilenet_weights() 
    
    num_samples = len(X_train) if not max_samples else min(len(X_train), max_samples)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Création d'un dictionnaire vide pour accumuler les gradients
        accumulated_grads = {k: np.zeros_like(v) for k, v in weights.items()}
        current_batch_size = 0

        for i, idx in enumerate(indices):
            x = X_train[idx]
            y = int(Y_train[idx])
            
            # --- C'EST ICI QU'ON FERA LA DATA AUGMENTATION ---
            x = augment_image_numpy(x)
            
            # Calcul des gradients pour cette image
            loss, preds, grads = compute_gradients(x, y, weights)
            total_loss += loss
            if np.argmax(preds) == y:
                correct += 1
                
            # On accumule les gradients
            for k in weights.keys():
                accumulated_grads[k] += grads[k]
            current_batch_size += 1

            # Si on a atteint la taille du batch (ou la fin du dataset), on met à jour les poids
            if current_batch_size == batch_size or i == num_samples - 1:
                for k in weights.keys():
                    # On met à jour avec la moyenne des gradients du batch
                    weights[k] -= learning_rate * (accumulated_grads[k] / current_batch_size)
                    # On réinitialise l'accumulateur pour le prochain batch
                    accumulated_grads[k].fill(0.0)
                current_batch_size = 0

            if (i+1) % max(10, num_samples // 4) == 0:
                print(f"  [Epoch {epoch+1}] Step {i+1}/{num_samples} | loss moyenne: {total_loss/(i+1):.4f}")
                
        epoch_time = time.time() - start_time
        print('-'*40)
        print(f"Epoch {epoch+1}/{epochs} — time {epoch_time:.2f}s — loss {total_loss/num_samples:.4f} — acc {correct/num_samples*100:.2f}%")
        
        # Réduction douce du learning rate
        learning_rate *= 0.9 
        
    return weights

# --- Dataset loader from archive CSV ---

def load_dataset_from_csv(csv_path, base_dir='archive', img_size=(96,96), max_samples=None):
    paths = []
    labels = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get('path') or row.get('image') or list(row.values())[0]
            l = row.get('label') or (list(row.values())[1] if len(row) > 1 else 0)
            full = os.path.join(base_dir, p)
            paths.append(full)
            labels.append(int(l))
            if max_samples and len(paths) >= max_samples:
                break
    X = []
    Y = []
    for p, l in zip(paths, labels):
        if not os.path.exists(p):
            # try without base_dir
            if os.path.exists(p):
                img_path = p
            else:
                continue
        else:
            img_path = p
        try:
            img = Image.open(img_path).convert('L').resize(img_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr.reshape((img_size[0], img_size[1], 1))
            X.append(arr)
            Y.append(int(l))
        except Exception:
            continue
    if len(X) == 0:
        return None, None
    return np.stack(X, axis=0), np.array(Y, dtype=np.int32)

# --- Main: load archive dataset and train ---
if __name__ == '__main__':
    train_csv = os.path.join('archive', 'train.csv')
    val_csv = os.path.join('archive', 'val.csv')
    print('Loading archive dataset...')
    if os.path.exists(train_csv):
        X_train, Y_train = load_dataset_from_csv(train_csv, base_dir='archive', img_size=(96,96), max_samples=250)
        print('Loaded', 0 if X_train is None else len(X_train), 'train samples')
    else:
        X_train, Y_train = None, None
    if X_train is None:
        print('Falling back to random dataset (100 samples)')
        X_train = np.random.rand(100, 96, 96, 1).astype(np.float32)
        Y_train = np.random.randint(0, 5, size=(100,))
    weights = train_mobilenet_from_scratch(X_train, Y_train, epochs=10, learning_rate=0.001, max_samples=250)
    print('\nExporting weights to .bin...')
    for name, arr in weights.items():
        arr.astype(np.float32).tofile(f'{name}.bin')
    print('Done.')
