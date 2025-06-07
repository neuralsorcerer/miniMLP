import numpy as np
from miniMLP.optimizers import SGD, Adam


def test_sgd_update():
    opt = SGD(learning_rate=0.1)
    w = np.array([[1.0, -1.0]])
    b = np.array([[0.5, -0.5]])
    grads = {'dW': np.array([[0.2, -0.2]]), 'db': np.array([[0.1, -0.1]])}
    opt.update(w, b, grads)
    assert np.allclose(w, np.array([[0.98, -0.98]]))
    assert np.allclose(b, np.array([[0.49, -0.49]]))


def test_adam_update():
    opt = Adam(learning_rate=0.1)
    w = np.array([[1.0, -1.0]])
    b = np.array([[0.0, 0.0]])
    grads = {'dW': np.array([[0.1, -0.1]]), 'db': np.array([[0.0, 0.0]])}
    opt.update(w, b, grads, t=1)
    # after first update, w should decrease slightly
    assert np.any(w != np.array([[1.0, -1.0]]))