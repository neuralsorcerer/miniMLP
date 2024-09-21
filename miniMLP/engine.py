import numpy as np
from miniMLP.layers import Layer
from miniMLP.losses import LossFunction
from miniMLP.optimizers import Optimizer

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

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through all layers."""
        for layer in self.layers:
            X = layer.forward(X, training=True)  # Enable training flag for dropout, etc.
        return X

    def backward(self, dA: np.ndarray) -> None:
        """Backpropagate error through all layers."""
        for layer in reversed(self.layers):
            dA = layer.backward(dA)  # No need for learning_rate here; handled by optimizer

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
              X_val: np.ndarray = None, Y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01,
              validation: bool = False, verbose: bool = True) -> dict:
        """
        Train the MLP with the given training data.
        
        Args:
            X_train: Training data (features).
            Y_train: Training labels (targets).
            X_val: Validation data (features).
            Y_val: Validation labels (targets).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches.
            learning_rate: Initial learning rate for the optimizer.
            validation: If True, perform validation after each epoch.
            verbose: If True, print the loss at each epoch.
            
        Returns:
            A dictionary with training and validation losses (if validation is enabled).
        """
        history = {'train_loss': [], 'val_loss': []}
        num_batches = X_train.shape[0] // batch_size

        for epoch in range(epochs):
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
                Y_pred = self.forward(X_batch)

                # Compute the loss
                loss = self.loss_function.compute_loss(Y_batch, Y_pred)
                epoch_loss += loss

                # Backward pass
                dA = self.loss_function.compute_gradient(Y_batch, Y_pred)
                self.backward(dA)

                # Update weights and biases using the optimizer
                for layer in self.layers:
                    self.optimizer.update(layer.weights, layer.biases, layer.grads)

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
