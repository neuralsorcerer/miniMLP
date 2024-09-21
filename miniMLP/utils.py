import numpy as np

def create_batches(X: np.ndarray, Y: np.ndarray, batch_size: int):
    """Create mini-batches from the dataset."""
    n = X.shape[0]
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    for i in range(0, n, batch_size):
        yield X_shuffled[i:i + batch_size], Y_shuffled[i:i + batch_size]
