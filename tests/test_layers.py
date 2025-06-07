import numpy as np
from miniMLP.layers import Layer
from miniMLP.activation import ActivationFunction
from miniMLP.regularizers import L2Regularizer


def test_forward_backward_shapes():
    layer = Layer(3, 4, ActivationFunction.relu)
    X = np.random.randn(5, 3)
    out = layer.forward(X)
    assert out.shape == (5, 4)
    dA = np.random.randn(5, 4)
    dX = layer.backward(dA)
    assert dX.shape == (5, 3)
    assert 'dW' in layer.grads and 'db' in layer.grads


def test_dropout():
    layer = Layer(2, 2, ActivationFunction.relu, dropout_rate=0.5)
    X = np.ones((4, 2))
    out_train = layer.forward(X, training=True)
    # dropout should create zeros in output and store mask
    assert np.any(out_train == 0)
    assert layer.dropout_mask is not None
    out_eval = layer.forward(X, training=False)
    # output in eval mode should not use dropout mask
    assert not np.array_equal(out_train, out_eval)


def test_regularization_grad():
    reg = L2Regularizer(0.1)
    layer = Layer(2, 3, ActivationFunction.relu, regularizer=reg)
    X = np.random.randn(4, 2)
    layer.forward(X)
    dA = np.ones((4, 3))
    layer.backward(dA)
    # gradient should include regularization term
    assert np.allclose(layer.grads['dW'], np.dot(X.T, dA * layer.activation_derivative(layer.Z)) + reg(layer.weights))


def test_xavier_init_variance():
    layer = Layer(4, 2, ActivationFunction.relu, init='xavier')
    var = np.var(layer.weights)
    expected = 1.0 / (4 + 2)
    assert abs(var - expected) < expected