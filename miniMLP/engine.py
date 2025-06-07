import numpy as np
from miniMLP.layers import Layer
from miniMLP.losses import LossFunction
from miniMLP.optimizers import Optimizer
import pickle

class MLP:
    """Multilayer Perceptron neural network."""
    
    def __init__(self, layers: list[Layer], loss_function: LossFunction, optimizer: Optimizer):
        """
        Initialize the MLP with layers, loss function, and optimizer.
        
        Args:
            layers: List of Layer objects (each fully connected layer).
            loss_function: The loss function object to calculate the error.
            optimizer: The optimizer object for updating weights and biases.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.iteration = 0

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward propagation through all layers."""
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X

    def backward(self, dA: np.ndarray) -> None:
        """Backpropagate error through all layers and collect gradients."""
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
              X_val: np.ndarray = None, Y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32,
              validation: bool = False, verbose: bool = True,
              lr_scheduler=None, early_stopping: bool = False,
              patience: int = 5, min_delta: float = 0.0,
              clip_value: float | None = None) -> dict:
        """
        Train the MLP with the given training data.
        
        Args:
            X_train: Training data (features).
            Y_train: Training labels (targets).
            X_val: Validation data (features).
            Y_val: Validation labels (targets).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches.
            validation: If True, perform validation after each epoch.
            verbose: If True, print the loss at each epoch.
            lr_scheduler: Optional callable returning the learning rate for a
                given epoch. If provided, it is called at the start of each
                epoch and the optimizer's learning rate is updated.
            early_stopping: If True, stop training when validation loss does
                not improve for ``patience`` epochs.
            patience: Number of epochs with no improvement after which
                training will be stopped when ``early_stopping`` is True.
            min_delta: Minimum change in the monitored quantity to qualify as
                an improvement.
            clip_value: If not ``None``, clip gradients to this absolute value
                before applying updates.
            
        Returns:
            A dictionary with training and validation losses (if validation is enabled).
        """
        history = {'train_loss': [], 'val_loss': []}
        num_batches = X_train.shape[0] // batch_size
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            if lr_scheduler is not None and hasattr(self.optimizer, 'learning_rate'):
                self.optimizer.learning_rate = lr_scheduler(epoch)

            epoch_loss = 0
            
            # Shuffle the training data at the start of each epoch
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]
            
            # Mini-batch training
            for batch in range(num_batches):
                X_batch = X_train_shuffled[batch * batch_size:(batch + 1) * batch_size]
                Y_batch = Y_train_shuffled[batch * batch_size:(batch + 1) * batch_size]

                # Forward pass
                Y_pred = self.forward(X_batch, training=True)

                # Compute the loss
                loss = self.loss_function.compute_loss(Y_batch, Y_pred)
                epoch_loss += loss

                # Backward pass
                dA = self.loss_function.compute_gradient(Y_batch, Y_pred)
                self.backward(dA)

                # Update weights and biases using the optimizer
                self.iteration += 1
                for layer in self.layers:
                    if clip_value is not None:
                        layer.grads['dW'] = np.clip(layer.grads['dW'], -clip_value, clip_value)
                        layer.grads['db'] = np.clip(layer.grads['db'], -clip_value, clip_value)
                    self.optimizer.update(layer.weights, layer.biases, layer.grads, t=self.iteration)

            # Average loss per batch
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

            # Validation (if enabled)
            if validation and X_val is not None and Y_val is not None:
                Y_val_pred = self.forward(X_val)
                val_loss = self.loss_function.compute_loss(Y_val, Y_val_pred)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [(layer.weights.copy(), layer.biases.copy()) for layer in self.layers]
                else:
                    patience_counter += 1
                    if early_stopping and patience_counter >= patience:
                        if verbose:
                            print("Early stopping triggered")
                        if best_weights is not None:
                            for layer, (w, b) in zip(self.layers, best_weights):
                                layer.weights = w
                                layer.biases = b
                        return history

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input data for prediction.
        
        Returns:
            Predicted values.
        """
        for layer in self.layers:
            X = layer.forward(X, training=False)  # Disable training flag for dropout, etc.
        return X
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Return discrete class predictions."""
        Y_pred = self.predict(X)
        if Y_pred.shape[1] == 1:
            return (Y_pred > 0.5).astype(int)
        return np.argmax(Y_pred, axis=1)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
        """Compute loss and accuracy on a dataset."""
        Y_pred = self.predict(X)
        loss = self.loss_function.compute_loss(Y, Y_pred)
        if Y_pred.shape[1] == 1:
            preds = (Y_pred > 0.5).astype(int)
            accuracy = np.mean(preds == Y)
        else:
            preds = np.argmax(Y_pred, axis=1)
            labels = np.argmax(Y, axis=1)
            accuracy = np.mean(preds == labels)
        return loss, accuracy

    def get_weights(self):
        """Return a copy of model weights and biases for all layers."""
        return [(layer.weights.copy(), layer.biases.copy()) for layer in self.layers]

    def set_weights(self, weights):
        """Set model weights and biases from a list of tuples."""
        for layer, (w, b) in zip(self.layers, weights):
            layer.weights = w.copy()
            layer.biases = b.copy()

    def save(self, filepath: str) -> None:
        """Save model weights to a file using pickle."""
        state = [
            {
                'weights': layer.weights,
                'biases': layer.biases,
            }
            for layer in self.layers
        ]
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        """Load model weights from a file created by :meth:`save`."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        for layer, params in zip(self.layers, state):
            layer.weights = params['weights']
            layer.biases = params['biases']

    def summary(self) -> None:
        """Print a summary of the network architecture."""
        print("Layer\tInput\tOutput\tActivation\tParams")
        total = 0
        for i, layer in enumerate(self.layers):
            params = layer.input_size * layer.output_size + layer.output_size
            total += params
            print(f"{i}\t{layer.input_size}\t{layer.output_size}\t{layer.activation.__name__}\t{params}")
        print(f"Total params: {total}")