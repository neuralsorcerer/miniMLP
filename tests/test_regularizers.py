import numpy as np
from miniMLP.regularizers import L1Regularizer, L2Regularizer


def test_l1_regularizer():
    reg = L1Regularizer(0.1)
    w = np.array([-1.0, 0.5])
    out = reg(w)
    assert np.allclose(out, 0.1 * np.sign(w))


def test_l2_regularizer():
    reg = L2Regularizer(0.2)
    w = np.array([-1.0, 0.5])
    out = reg(w)
    assert np.allclose(out, 0.2 * w)