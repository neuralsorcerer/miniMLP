from typing import Optional
import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def update(self, weights: np.ndarray, biases: np.ndarray, grads: dict, t: Optional[int] = None):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum and Nesterov acceleration."""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights: np.ndarray, biases: np.ndarray, grads: dict, t: Optional[int] = None) -> None:
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)

        if self.momentum > 0:
            self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * grads['dW']
            self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * grads['db']
            
            if self.nesterov:
                weights += self.momentum * self.velocity_w - self.learning_rate * grads['dW']
                biases += self.momentum * self.velocity_b - self.learning_rate * grads['db']
            else:
                weights += self.velocity_w
                biases += self.velocity_b
        else:
            weights -= self.learning_rate * grads['dW']
            biases -= self.learning_rate * grads['db']


class Momentum(Optimizer):
    """Momentum optimizer."""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights: np.ndarray, biases: np.ndarray, grads: dict, t: Optional[int] = None) -> None:
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)

        # Update the velocities
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * grads['dW']
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * grads['db']

        # Update the weights and biases
        weights += self.velocity_w
        biases += self.velocity_b


class NAG(SGD):
    """Nesterov Accelerated Gradient (NAG) optimizer, inheriting from SGD."""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate=learning_rate, momentum=momentum, nesterov=True)


class Adam(Optimizer):
    """Adam optimizer."""
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, weights: np.ndarray, biases: np.ndarray, grads: dict, t: int) -> None:
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads['dW']
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads['dW'] ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** t)

        # Update weights and biases
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        biases -= self.learning_rate * grads['db']


class RMSProp(Optimizer):
    """RMSProp optimizer."""
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.s = None

    def update(self, weights: np.ndarray, biases: np.ndarray, grads: dict, t: Optional[int] = None) -> None:
        if self.s is None:
            self.s = np.zeros_like(weights)

        # Update the running average of squared gradients
        self.s = self.rho * self.s + (1 - self.rho) * (grads['dW'] ** 2)

        # Update weights and biases
        weights -= self.learning_rate * grads['dW'] / (np.sqrt(self.s) + self.epsilon)
        biases -= self.learning_rate * grads['db']
