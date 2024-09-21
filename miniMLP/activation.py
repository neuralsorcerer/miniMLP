import numpy as np

class ActivationFunction:
    """Collection of common activation functions used in neural networks."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation function."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
        return x * cdf

    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        """Softplus activation function."""
        return np.log(1 + np.exp(x))

    @staticmethod
    def selu(x: np.ndarray, alpha: float = 1.67326, scale: float = 1.0507) -> np.ndarray:
        """SELU activation function."""
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def prelu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """PReLU activation function."""
        return np.maximum(alpha * x, x)

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation function."""
        return x * ActivationFunction.sigmoid(x)

    @staticmethod
    def gaussian(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
        """Gaussian activation function."""
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
