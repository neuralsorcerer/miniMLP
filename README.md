# miniMLP

This repository contains an implementation of a small Neural Network in Python, with the ability to specify the number of hidden layers, the number of neurons in each layer, and the activation function, and the optimizer to be used.

## Installation

```
pip install miniMLP
```

## Dependencies

The code requires the following dependencies:
- NumPy
- matplotlib (for visualization)

## Example Usage

Create an instance of the **`MLP`** class, specifying the input size, output size, number of hidden layers, sizes of each hidden layer, and activation function.
```python
import numpy as np
from miniMLP.engine import MLP

mlp = MLP(input_size=2, output_size=1, hidden_layers=3, hidden_sizes=[4, 6, 4], activation='relu')
```
Then, train the MLP using the **`train`** method, providing the training data (an array of inputs and corresponding labels), and optional parameters such as the learning rate, number of epochs, and optimizer.
```python
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

mlp.train(X_train, y_train, epochs=2000, learning_rate=0.0001, optimizer='Adam')
```
Finally, use the **`predict`** method to obtain predictions for new data points.
```python
y_pred = mlp.predict(X_new)
```

## Activation Functions

The following activation functions are currently supported:
- Sigmoid
- ReLU
- Tanh
- Softmax
- Leaky ReLU
- ELU
- GELU
- Softplus
- SeLU
- PReLu
- Swish
- Gaussian

## Optimizers

The following optimizers are currently supported:
- Adam
- Stochastic Gradient Descent
- Momentum

## License

This project is licensed under the MIT License. Feel free to use and modify this code for your own projects.
