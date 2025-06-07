import math

class StepLR:
    """Step learning rate scheduler."""
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        factor = self.gamma ** (epoch // self.step_size)
        return self.initial_lr * factor


class ExponentialLR:
    """Exponential learning rate scheduler."""
    def __init__(self, initial_lr: float, gamma: float = 0.99):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** epoch)