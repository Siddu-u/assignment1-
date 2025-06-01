**Question 1: Tensor Manipulations & Reshaping**

Objective:

To understand the fundamentals of tensor operations such as creating, reshaping, transposing, and broadcasting using TensorFlow.

Steps and Theoretical Explanation:

1. **Create a Random Tensor of Shape (4, 6)**

   * A tensor is a multi-dimensional array. TensorFlow provides functions like `tf.random.uniform` to create random tensors.
   * Shape `(4, 6)` means 4 rows and 6 columns, totaling 24 elements.

2. **Find its Rank and Shape**

   * **Rank** refers to the number of dimensions (e.g., a matrix has rank 2).
   * **Shape** gives the size along each dimension, which can be obtained using `tf.shape()` and `tf.rank()`.

3. **Reshape to (2, 3, 4)**

   * Reshaping rearranges data into a new shape without changing the underlying values.
   * The new shape must have the same number of total elements: `2 × 3 × 4 = 24`.

4. **Transpose to (3, 2, 4)**

   * Transposing reorders dimensions. Here, we change the order of axes using `tf.transpose(tensor, perm=[1, 0, 2])`.

5. **Broadcasting a Smaller Tensor (1, 4) and Adding**

   * Broadcasting allows TensorFlow to automatically expand a tensor of lower rank to match another tensor’s shape during arithmetic operations.
   * A tensor of shape `(1, 4)` can be added to a tensor of shape `(3, 2, 4)` because TensorFlow will replicate the smaller tensor across the missing dimensions.

Expected Output:

* Print the original tensor's rank and shape.
* Show new shapes after reshaping and transposing.
* Show the result after broadcasting and addition.

**Question 2: Loss Functions & Hyperparameter Tuning**

Objective:

To understand and compare loss functions and observe their sensitivity to prediction changes.

Steps and Theoretical Explanation:

1. **Define `y_true` and `y_pred`**

   * These are true labels and model predictions.
   * Typically, they are probability distributions for classification problems.

2. **Compute MSE and CCE**

   * **Mean Squared Error (MSE):** Measures average of squared differences. Good for regression tasks.
   * **Categorical Cross-Entropy (CCE):** Measures how far predicted probabilities are from true one-hot vectors. Ideal for classification.

3. **Modify Predictions and Observe Loss Changes**

   * Change `y_pred` slightly and observe how MSE and CCE respond.
   * MSE responds linearly; CCE can grow sharply if the model becomes confident in the wrong prediction.

4. **Plot the Loss Values Using Matplotlib**

   * Create a bar chart to visualize the comparison between MSE and CCE for different predictions.

Expected Output:

* Print the computed MSE and CCE values for different predictions.
* Visualize them in a bar chart using Matplotlib.

**Question 3: Train a Neural Network and Log to TensorBoard**
Objective:

To train a neural network using the MNIST dataset and visualize metrics using TensorBoard.

Steps and Theoretical Explanation:

1. **Load and Preprocess the MNIST Dataset**

   * MNIST consists of 28×28 grayscale images of handwritten digits (0–9).
   * Data is normalized (pixel values scaled to \[0, 1]) for better model performance.

2. **Train a Neural Network**

   * A simple feed-forward neural network includes:

     * `Flatten` layer to convert 2D image to 1D.
     * `Dense` layer with ReLU activation.
     * `Dropout` layer for regularization.
     * Output layer with `softmax` for multi-class classification.
   * Compile with `Adam` optimizer and train for 5 epochs.

3. **Enable TensorBoard Logging**

   * Use `tf.keras.callbacks.TensorBoard` to save training logs.
   * Logs are written to `logs/fit/`, which can be visualized with TensorBoard.

4. **Launch TensorBoard and Analyze Trends**

   * Run `tensorboard --logdir=logs/fit` in terminal.
   * Open `http://localhost:6006` in a browser.
   * Observe training and validation curves.

Expected Output:

* Model trains for 5 epochs.
* Logs are stored in `logs/fit/`.
* Training and validation metrics (loss, accuracy) are visualized in TensorBoard.

