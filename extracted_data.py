import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

folder_path = 'extracted_parameters'

parameters = []
all_files = sorted(os.listdir(folder_path))
first_set_files = all_files[:10]  

for filename in first_set_files:
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        layer_params = np.load(file_path)
        parameters.append(layer_params)
        print(f"Loaded parameters from {filename}, shape: {layer_params.shape}")

model = create_model()

try:
    model.set_weights(parameters)
    print("Model weights set successfully.")
except ValueError as e:
    print(f"Error setting weights: {e}")

input_image = tf.Variable(tf.random.normal([1, 28, 28, 1]))  
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for step in range(1000):
    with tf.GradientTape() as tape:
        output = model(input_image)
        loss = -tf.reduce_max(output)  

    gradients = tape.gradient(loss, input_image)
    optimizer.apply_gradients([(gradients, input_image)])

input_image_np = input_image.numpy().squeeze()
plt.imshow(input_image_np, cmap='gray')
plt.title("Reconstructed Image")
plt.show()
