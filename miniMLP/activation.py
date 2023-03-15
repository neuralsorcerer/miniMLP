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
    
    def gelu(x):
        cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
        return x * cdf

    def softplus(x):
        return np.log(1 + np.exp(x))

    def selu(x, alpha=1.67326, scale=1.0507):
        mask = (x > 0)
        out = scale * (mask * x + (1 - mask) * alpha * (np.exp(x) - 1))
        return out

    def prelu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def swish(x):
        return x * ActivationFunction.sigmoid(x)

    def gaussian(x, mu=0, sigma=1):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))



