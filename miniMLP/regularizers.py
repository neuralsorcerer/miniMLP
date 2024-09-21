import numpy as np

class L2Regularizer:
    """L2 Regularization, also known as weight decay."""
    def __init__(self, alpha: float):
        """
        Initialize the L2 regularizer.
        
        Args:
            alpha: Regularization strength (penalty term).
        """
        self.alpha = alpha

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        """Apply L2 regularization to the weights."""
        return self.alpha * weights

class Dropout:
    """Dropout regularization technique."""
    def __init__(self, rate: float):
        """
        Initialize dropout regularizer.
        
        Args:
            rate: The dropout rate (probability of dropping a unit).
        """
        self.rate = rate

    def apply(self, A: np.ndarray) -> np.ndarray:
        """Apply dropout to the activations during training."""
        mask = np.random.binomial(1, 1 - self.rate, size=A.shape)
        return A * mask / (1 - self.rate)
