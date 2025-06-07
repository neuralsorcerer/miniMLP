import numpy as np
from miniMLP.engine import MLP
from miniMLP.layers import Layer
from miniMLP.activation import ActivationFunction
from miniMLP.losses import MSE, BinaryCrossEntropy
from miniMLP.optimizers import SGD
import tempfile
import os


def build_xor_mlp():
    layers = [
        Layer(2, 8, ActivationFunction.relu),
        Layer(8, 1, ActivationFunction.sigmoid)
    ]
    return MLP(layers, MSE(), SGD(0.5))


def test_train_and_evaluate():
    mlp = build_xor_mlp()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    history = mlp.train(X, Y, epochs=4000, batch_size=4, validation=False, verbose=False, clip_value=1.0)
    loss, acc = mlp.evaluate(X, Y)
    assert loss < 0.1
    assert acc > 0.9


def test_early_stopping():
    mlp = build_xor_mlp()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    history = mlp.train(X, Y, X, Y, epochs=50, batch_size=4, validation=True, verbose=False, early_stopping=True, patience=2)
    assert len(history['train_loss']) <= 50


def test_save_and_load():
    mlp = build_xor_mlp()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    mlp.train(X, Y, epochs=50, batch_size=4, validation=False, verbose=False)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    mlp.save(path)
    mlp2 = build_xor_mlp()
    mlp2.load(path)
    os.remove(path)
    p1 = mlp.predict(X)
    p2 = mlp2.predict(X)
    assert np.allclose(p1, p2)


def test_get_set_weights():
    mlp = build_xor_mlp()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    mlp.train(X, Y, epochs=10, batch_size=4, verbose=False)
    weights = mlp.get_weights()
    pred1 = mlp.predict(X)
    # corrupt weights
    mlp.layers[0].weights += 1.0
    mlp.set_weights(weights)
    pred2 = mlp.predict(X)
    assert np.allclose(pred1, pred2)


def test_predict_classes():
    mlp = build_xor_mlp()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    mlp.train(X, Y, epochs=4000, batch_size=4, verbose=False)
    preds = mlp.predict_classes(X)
    assert np.array_equal(preds, Y)


def test_binary_cross_entropy_training():
    layers = [
        Layer(2, 8, ActivationFunction.relu),
        Layer(8, 1, ActivationFunction.sigmoid)
    ]
    mlp = MLP(layers, BinaryCrossEntropy(), SGD(0.5))
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    mlp.train(X, Y, epochs=4000, batch_size=4, verbose=False)
    loss, acc = mlp.evaluate(X, Y)
    assert loss < 0.1
    assert acc > 0.9