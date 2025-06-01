Deep Learning Tasks: Tensor Manipulations, Loss Functions, and TensorBoard Logging

Task 1: Tensor Manipulations & Reshaping

Ojective:
Perform tensor reshaping, transposing, broadcasting, and understand broadcasting in TensorFlow.

Code:
import tensorflow as tf

# 1. Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", tensor)

# 2. Rank and shape
rank = tf.rank(tensor)
shape = tf.shape(tensor)
print("\nRank:", rank.numpy())
print("Shape:", shape.numpy())

# 3. Reshape to (2, 3, 4) and transpose to (3, 2, 4)
reshaped = tf.reshape(tensor, (2, 3, 4))
transposed = tf.transpose(reshaped, perm=[1, 0, 2])
print("\nReshaped Tensor (2, 3, 4):\n", reshaped)
print("\nTransposed Tensor (3, 2, 4):\n", transposed)

# 4. Broadcast a (1, 4) tensor
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
result = transposed + small_tensor
print("\nResult after broadcasting:\n", result)

Explanation:

Broadcasting in TensorFlow automatically adjusts the shape of tensors during operations. A tensor of shape `(1, 4)` can be added to one with shape `(3, 2, 4)` by expanding it to `(3, 2, 4)` implicitly.

Task 2: Loss Functions & Hyperparameter Tuning

Objective:

Compare Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses for different model predictions.

Code:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Define true values and predictions
y_true = tf.constant([[0, 0, 1]], dtype=tf.float32)
y_pred_1 = tf.constant([[0.1, 0.1, 0.8]])
y_pred_2 = tf.constant([[0.3, 0.4, 0.3]])

# 2. Compute losses
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

mse_1 = mse(y_true, y_pred_1).numpy()
cce_1 = cce(y_true, y_pred_1).numpy()
mse_2 = mse(y_true, y_pred_2).numpy()
cce_2 = cce(y_true, y_pred_2).numpy()

print(f"Prediction 1 - MSE: {mse_1:.4f}, CCE: {cce_1:.4f}")
print(f"Prediction 2 - MSE: {mse_2:.4f}, CCE: {cce_2:.4f}")

# 3. Plotting
labels = ['Prediction 1', 'Prediction 2']
mse_values = [mse_1, mse_2]
cce_values = [cce_1, cce_2]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, mse_values, width, label='MSE', color='skyblue')
plt.bar(x + width/2, cce_values, width, label='CCE', color='salmon')
plt.ylabel('Loss Value')
plt.title('Loss Comparison')
plt.xticks(x, labels)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

Output:

* Loss values printed in the console
* A bar chart comparing MSE vs CCE for two prediction cases


Task 3: Train a Neural Network and Log to TensorBoard

Objective:

* Train a simple neural network on the MNIST dataset and log metrics for TensorBoard visualization.

Code:

import tensorflow as tf
import datetime

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_cb])
```

Launch TensorBoard:


4.1 Questions to answer

1. What patterns do you observe in the training and validation accuracy curves?

* Training accuracy usually improves with each epoch.
* Validation accuracy may increase initially, then plateau or drop if overfitting occurs.

2. How can you use TensorBoard to detect overfitting?

* If training accuracy keeps increasing but validation accuracy drops or stagnates, it indicates overfitting.
* Similarly, validation loss rising while training loss drops is another sign.

3. What happens when you increase the number of epochs?

* The model may start overfitting after a point.
* Early improvements in accuracy may flatten out.
* Validation performance may degrade if training continues excessively.

