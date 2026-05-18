"""Inference script for MNIST TFLite model.

Usage:
  python script_inference.py --model mnist_custom.tflite --samples 20
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_mnist_random(n):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = x_test.astype(np.float32)
    y = y_test
    idx = np.random.choice(len(x), size=n, replace=False)
    imgs = x[idx]
    labels = y[idx]
    # Normalize to [0,1]. Leave original shape intact.
    imgs = imgs / 255.0
    return imgs, labels

def run_inference(model_path, samples=20):
    print('Loading TFLite model from', model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details['index']
    output_index = output_details['index']
    
    # Extract the exact shape the model expects (usually [1, 28, 28, 1] or [1, 784])
    expected_shape = input_details['shape']
    dtype = input_details['dtype']
    
    # Handle quantization scale/zero_point if model is int8 or uint8
    scale, zero_point = input_details.get('quantization', (1.0, 0))
    if scale == 0:
        scale = 1.0

    images, labels = load_mnist_random(samples)
    correct_predictions = 0
    print_results = []

    # Run inference ONE image at a time (Standard TFLite practice)
    for i in range(samples):
        # 1. Isolate one image and reshape it to the model's exact expected shape
        img = images[i]
        img_reshaped = img.reshape(expected_shape)

        # 2. Apply quantization mapping if necessary
        if dtype == np.uint8 or dtype == np.int8:
            inp = (img_reshaped / scale + zero_point).astype(dtype)
        else:
            inp = img_reshaped.astype(dtype)

        # 3. Set tensor, invoke, and get results
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        # 4. Calculate predictions
        # Note: output_data shape is usually [1, num_classes]
        pred = np.argmax(output_data, axis=1)[0]
        if pred == labels[i]:
            correct_predictions += 1

        # Save first 10 for detailed output
        if i < 10:
            probs = output_data[0]
            top = np.argsort(probs)[-3:][::-1]
            print_results.append(f'Idx {i}: label={labels[i]}  pred={pred}  top3={list(zip(top, probs[top].round(4)))}')

    # Final Output
    acc = correct_predictions / samples
    print(f'\nSamples: {samples}  Accuracy: {acc*100:.2f}%')
    print('-' * 40)
    for res in print_results:
        print(res)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default='mnist_custom.tflite', help='Path to .tflite model')
    p.add_argument('--samples', '-n', type=int, default=50, help='Number of random MNIST samples to test')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_inference(args.model, args.samples)