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
