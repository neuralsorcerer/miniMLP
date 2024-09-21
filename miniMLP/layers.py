import numpy as np
from typing import Callable, Optional
from miniMLP.regularizers import L2Regularizer, Dropout

class Layer:
    """Represents a fully connected layer in the neural network."""

    def __init__(self, input_size: int, output_size: int, activation: Callable, 
                 regularizer: Optional[L2Regularizer] = None, dropout_rate: float = 0.0):
        """
        Initialize the layer with weights, biases, and optional regularization/dropout.
        
        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
            activation: Activation function (e.g., sigmoid, relu).
            regularizer: Optional regularizer to apply (e.g., L2Regularizer).
            dropout_rate: Dropout rate, default is 0 (no dropout).
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the layer.
        
        Args:
            X: Input data or activations from the previous layer.
            training: Boolean flag to indicate if it's training mode (applies dropout).
        
        Returns:
            A: The activation after applying weights, biases, and activation function.
        """
        Z = np.dot(X, self.weights) + self.biases
        A = self.activation(Z)

        # Apply dropout if in training mode
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=A.shape) / (1 - self.dropout_rate)
            A *= self.dropout_mask

        return A

    def backward(self, dA: np.ndarray, X: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backpropagation through the layer.
        
        Args:
            dA: Gradient of the loss function with respect to the output of this layer.
            X: Input to this layer (activations from the previous layer or input data).
            learning_rate: Learning rate for weight updates.
        
        Returns:
            dA_prev: Gradient of the loss function with respect to the input of this layer (to pass to previous layer).
        """
        # Compute the derivative of the activation function
        dZ = dA * self.activation_derivative(X)

        # Compute gradients with respect to weights and biases
        dW = np.dot(X.T, dZ) + (self.regularizer(self.weights) if self.regularizer else 0)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        # Backpropagate dropout effect if dropout was applied
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            dZ *= self.dropout_mask

        # Return gradient for the previous layer
        return np.dot(dZ, self.weights.T)

    def activation_derivative(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.
        
        Args:
            X: Input data for this layer.
        
        Returns:
            Derivative of the activation function with respect to X.
        """
        return self.activation(X) * (1 - self.activation(X)) 
