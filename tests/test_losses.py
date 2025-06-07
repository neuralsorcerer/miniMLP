import numpy as np
from miniMLP.losses import MSE, MAE, CrossEntropy, BinaryCrossEntropy, HingeLoss, HuberLoss


def numeric_grad(loss_fn, y_true, y_pred, eps=1e-6):
    grad = np.zeros_like(y_pred)
    flat_pred = y_pred.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_pred.size):
        orig = flat_pred[i]
        flat_pred[i] = orig + eps
        loss_plus = loss_fn.compute_loss(y_true, y_pred)
        flat_pred[i] = orig - eps
        loss_minus = loss_fn.compute_loss(y_true, y_pred)
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        flat_pred[i] = orig
    return grad


def test_mse():
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.8, 0.2]])
    loss = MSE().compute_loss(y_true, y_pred)
    assert np.isclose(loss, ((0.2**2 + 0.2**2)/2))
    grad = MSE().compute_gradient(y_true, y_pred)
    num = numeric_grad(MSE(), y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)


def test_mae():
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.8, 0.2]])
    loss = MAE().compute_loss(y_true, y_pred)
    assert np.isclose(loss, (0.2 + 0.2)/2)
    grad = MAE().compute_gradient(y_true, y_pred)
    num = numeric_grad(MAE(), y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)


def test_cross_entropy():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    ce = CrossEntropy()
    grad = ce.compute_gradient(y_true, y_pred)
    num = numeric_grad(ce, y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)


def test_binary_cross_entropy():
    y_true = np.array([[1.0], [0.0], [1.0]])
    y_pred = np.array([[0.9], [0.2], [0.8]])
    bce = BinaryCrossEntropy()
    grad = bce.compute_gradient(y_true, y_pred)
    num = numeric_grad(bce, y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)


def test_hinge_loss():
    y_true = np.array([[1.0], [-1.0]])
    y_pred = np.array([[0.3], [-0.5]])
    hl = HingeLoss()
    grad = hl.compute_gradient(y_true, y_pred)
    num = numeric_grad(hl, y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)


def test_huber_loss():
    y_true = np.array([[0.0], [2.0]])
    y_pred = np.array([[1.0], [2.5]])
    huber = HuberLoss(delta=1.0)
    grad = huber.compute_gradient(y_true, y_pred)
    num = numeric_grad(huber, y_true, y_pred)
    assert np.allclose(grad, num, atol=1e-6)