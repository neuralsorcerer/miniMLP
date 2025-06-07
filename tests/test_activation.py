import numpy as np
from miniMLP.activation import ActivationFunction


def numeric_derivative(func, x, eps=1e-6):
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def test_sigmoid_derivative():
    x = np.linspace(-2, 2, 5)
    num = numeric_derivative(ActivationFunction.sigmoid, x)
    ana = ActivationFunction.sigmoid_derivative(x)
    assert np.allclose(num, ana, atol=1e-5)


def test_relu_derivative():
    x = np.array([-1.0, 1.0])
    num = numeric_derivative(ActivationFunction.relu, x)
    ana = ActivationFunction.relu_derivative(x)
    assert np.allclose(num, ana, atol=1e-5)


def test_gelu_derivative():
    x = np.array([-1.0, 0.0, 1.0])
    num = numeric_derivative(ActivationFunction.gelu, x)
    ana = ActivationFunction.gelu_derivative(x)
    assert np.allclose(num, ana, atol=1e-5)