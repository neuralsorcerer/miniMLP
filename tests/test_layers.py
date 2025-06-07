import numpy as np
from miniMLP.layers import Layer
from miniMLP.activation import ActivationFunction
from miniMLP.regularizers import L2Regularizer
from miniMLP.layers import BatchNormalization


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
    layer.weights[:] = 1.0  # ensure positive activations so dropout has effect
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


def test_batchnorm_forward_backward():
    bn = BatchNormalization(3)
    X = np.random.randn(5, 3)
    out = bn.forward(X, training=True)
    assert out.shape == (5, 3)
    dY = np.random.randn(5, 3)
    dX = bn.backward(dY)
    assert dX.shape == (5, 3)
    assert 'dW' in bn.grads and 'db' in bn.grads


def test_batchnorm_running_stats():
    bn = BatchNormalization(2, momentum=0.5)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    bn.forward(X, training=True)
    expected = X.mean(axis=0, keepdims=True) * 0.5
    assert np.allclose(bn.running_mean, expected)