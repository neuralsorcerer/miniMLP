# miniMLP

This repository contains an implementation of a small Multilayer Perceptron (MLP) Neural Network in Python,
with the ability to specify the number of hidden layers, the number of neurons in each layer, the activation function,
the optimizer to be used, and support for batch training, validation, and different loss functions.

## Installation

To install miniMLP, simply run:

```
pip install miniMLP
```

## Dependencies

The code requires the following dependencies:

- **NumPy**: For matrix operations and mathematical computations.
- **matplotlib** (optional): For visualization of training metrics, like loss curves.

Install them via pip:

```
pip install numpy matplotlib
```

## Features

- **Customizable Layers**: Specify the number of layers, neurons per layer, and activation functions.
- **Multiple Optimizers**: Choose from various optimizers (SGD, Adam, Momentum, etc.).
- **Flexible Loss Functions**: Support for several loss functions, including MSE, Cross Entropy, MAE, and more.
- **Mini-Batch Training**: Efficient training using mini-batches for large datasets.
- **Validation Support**: Monitor validation performance during training.
- **Training History**: Track loss over epochs, useful for plotting and debugging.
- **Learning Rate Scheduling**: Support for dynamic learning rates.

## Example Usage

### Creating an MLP Model

Create an instance of the **`MLP`** class by specifying the input size, output size, hidden layers, the number of neurons in each layer,
activation functions, and optimizer:

```python
import numpy as np
from miniMLP.engine import MLP
from miniMLP.activations import ActivationFunction
from miniMLP.optimizers import Adam
from miniMLP.losses import MSE
from miniMLP.layers import Layer

# Example MLP Architecture
layers = [
    Layer(input_size=2, output_size=4, activation=ActivationFunction.relu),
    Layer(input_size=4, output_size=6, activation=ActivationFunction.relu),
    Layer(input_size=6, output_size=1, activation=ActivationFunction.sigmoid)
]

# Define loss function and optimizer
loss_fn = MSE()
optimizer = Adam(learning_rate=0.001)

# Initialize MLP
mlp = MLP(layers=layers, loss_function=loss_fn, optimizer=optimizer)
```

### Training the MLP

Use the **`train`** method to train the MLP, specifying the training data, validation data, learning rate,
number of epochs, batch size, and more.

```python
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Train the model
mlp.train(X_train, y_train, epochs=2000, learning_rate=0.0001, batch_size=4)
```

### Making Predictions

After training, use the **`predict`** method to generate predictions for new data:

```python
X_new = np.array([[1, 1], [0, 0]])
y_pred = mlp.predict(X_new)
print(y_pred)
```

## Activation Functions

The following activation functions are supported:

- Sigmoid
- ReLU
- Tanh
- Softmax
- Leaky ReLU
- ELU
- GELU
- Softplus
- SeLU
- PReLU
- Swish
- Gaussian

## Optimizers

The following optimizers are supported:

- Adam
- Stochastic Gradient Descent (SGD) with momentum and Nesterov
- RMSProp
- Momentum
- Nesterov Accelerated Gradient (NAG)

## Loss Functions

Supported loss functions include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross Entropy (for classification)
- Binary Cross Entropy
- Hinge Loss (used in SVM)
- Huber Loss (robust regression)

## Example with Validation

You can also pass validation data to track model performance:

```python
X_val = np.array([[1, 1], [0, 1]])
y_val = np.array([[0], [1]])

history = mlp.train(X_train, y_train, X_val=X_val, Y_val=y_val, epochs=2000, learning_rate=0.0001, batch_size=4, validation=True)
```

This will output the training loss and validation loss for each epoch.

## Plotting Training and Validation Loss

If you track loss history during training, you can plot it using `matplotlib`:

```python
import matplotlib.pyplot as plt

plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify this code for your own projects.
