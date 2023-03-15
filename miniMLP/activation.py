import numpy as np

class ActivationFunction:

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        return np.maximum(0, x)
    
    def tanh(x):
        return np.tanh(x)

    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))


