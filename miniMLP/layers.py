import numpy as np
from typing import Callable, Optional
from miniMLP.activation import ActivationFunction
from miniMLP.regularizers import L2Regularizer

class Layer:
    """Represents a fully connected layer in the neural network."""

    def __init__(self, input_size: int, output_size: int, activation: Callable,
                 activation_derivative: Optional[Callable] = None,
                 regularizer: Optional[L2Regularizer] = None,
                 dropout_rate: float = 0.0,
                 init: str = "he"):
        """Initialize layer parameters.
        
        Args:
            input_size: Number of input features.
            output_size: Number of neurons in the layer.
            activation: Activation function to apply.
            activation_derivative: Optional derivative of the activation
                function. If ``None`` it is looked up by name in
                :class:`ActivationFunction`.
            regularizer: Optional regularizer to apply to the weights.
            dropout_rate: Probability of dropping a unit during training.
            init: Weight initialization method ("he" or "xavier").
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        if activation_derivative is None:
            derivative_name = f"{activation.__name__}_derivative"
            activation_derivative = getattr(ActivationFunction, derivative_name, None)
            if activation_derivative is None:
                raise ValueError(f"Derivative for activation {activation.__name__} not implemented")
        self.activation_derivative = activation_derivative

        # Parameters
        if init == "he":
            scale = np.sqrt(2.0 / input_size)
        elif init == "xavier":
            scale = np.sqrt(1.0 / (input_size + output_size))
        else:
            scale = 0.01
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.input = None
        self.Z = None
        self.A = None
        self.grads = {}

    def param_count(self) -> int:
        """Return the number of trainable parameters in this layer."""
        return self.input_size * self.output_size + self.output_size

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the layer.
        
        Args:
            X: Input data or activations from the previous layer.
            training: Boolean flag to indicate if it's training mode (applies dropout).
        
        Returns:
            A: The activation after applying weights, biases, and activation function.
        """
        self.input = X
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self.activation(self.Z)

        # Apply dropout if in training mode
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.A.shape) / (1 - self.dropout_rate)
            self.A *= self.dropout_mask

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backpropagation through the layer.
        
        Args:
            dA: Gradient of the loss with respect to the output of this layer.
        
        Returns:
            dA_prev: Gradient of the loss function with respect to the input of this layer (to pass to previous layer).
        """
        # Compute the derivative of the activation function
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            dA *= self.dropout_mask

        dZ = dA * self.activation_derivative(self.Z)

        # Compute gradients with respect to weights and biases
        dW = np.dot(self.input.T, dZ)
        if self.regularizer:
            dW += self.regularizer(self.weights)
        db = np.sum(dZ, axis=0, keepdims=True)
        self.grads = {'dW': dW, 'db': db}

        return np.dot(dZ, self.weights.T)


class BatchNormalization:
    """Batch Normalization layer."""

    def __init__(self, input_size: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        # Alias for optimizer compatibility
        self.weights = self.gamma
        self.biases = self.beta

        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))

        self.cache = None
        self.grads = {}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_mean = X.mean(axis=0, keepdims=True)
            batch_var = X.var(axis=0, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.cache = (X, batch_mean, batch_var, X_norm)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        return out

    def backward(self, dY: np.ndarray) -> np.ndarray:
        X, mean, var, X_norm = self.cache
        m = X.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        dgamma = np.sum(dY * X_norm, axis=0, keepdims=True)
        dbeta = np.sum(dY, axis=0, keepdims=True)

        dX_norm = dY * self.gamma
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * (X - mean), axis=0, keepdims=True)
        dX = dX_norm * std_inv + dvar * 2.0 * (X - mean) / m + dmean / m

        self.grads = {'dW': dgamma, 'db': dbeta}
        return dX

    def param_count(self) -> int:
        return 2 * self.input_size