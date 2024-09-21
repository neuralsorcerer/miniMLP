import numpy as np

class LossFunction:
    """Base class for all loss functions."""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the loss value between true and predicted values."""
        raise NotImplementedError

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss function."""
        raise NotImplementedError

class MSE(LossFunction):
    """Mean Squared Error loss"""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Mean Squared Error."""
        return np.mean(np.power(Y_true - Y_pred, 2))

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the MSE loss."""
        return 2 * (Y_pred - Y_true) / Y_true.size
    

class MAE(LossFunction):
    """Mean Absolute Error loss"""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Mean Absolute Error."""
        return np.mean(np.abs(Y_true - Y_pred))

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the MAE loss."""
        return np.sign(Y_pred - Y_true) / Y_true.size

class CrossEntropy(LossFunction):
    """Cross Entropy loss"""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Cross Entropy loss."""
        m = Y_pred.shape[0]
        return -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Cross Entropy loss."""
        return -(Y_true / (Y_pred + 1e-8))

class BinaryCrossEntropy(LossFunction):
    """Binary Cross Entropy loss"""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Binary Cross Entropy loss."""
        return -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8))

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Binary Cross Entropy loss."""
        return -(Y_true / (Y_pred + 1e-8)) + (1 - Y_true) / (1 - Y_pred + 1e-8)

class HingeLoss(LossFunction):
    """Hinge loss"""
    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Hinge loss."""
        return np.mean(np.maximum(0, 1 - Y_true * Y_pred))

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Hinge loss."""
        grad = np.zeros_like(Y_pred)
        mask = Y_true * Y_pred < 1
        grad[mask] = -Y_true[mask]
        return grad

class HuberLoss(LossFunction):
    """Huber loss"""
    def __init__(self, delta: float = 1.0):
        """
        Initialize the Huber loss.
        
        Args:
            delta: The threshold at which the loss transitions from quadratic to linear.
        """
        self.delta = delta

    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute the Huber loss."""
        error = Y_true - Y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.power(error, 2)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def compute_gradient(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Huber loss."""
        error = Y_pred - Y_true
        is_small_error = np.abs(error) <= self.delta
        return np.where(is_small_error, error, self.delta * np.sign(error))
