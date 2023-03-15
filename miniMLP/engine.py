import numpy as np
from miniMLP.activation import ActivationFunction

class MLP:
    def __init__(self, input_size, output_size, hidden_layers=1, hidden_sizes=[1], activation='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        # Initialize weights and biases for input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        # Initialize weights and biases for the rest of the hidden layers
        for i in range(hidden_layers - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))

        # Initialize weights and biases for last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def activation_func(self, x):
        if self.activation == 'sigmoid':
            return ActivationFunction.sigmoid(x)
        elif self.activation == 'relu':
            return ActivationFunction.relu(x)
        elif self.activation == 'tanh':
            return ActivationFunction.tanh(x)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax(x)
        elif self.activation == 'leaky_relu':
            return ActivationFunction.leaky_relu(x)
        elif self.activation == 'elu':
            return ActivationFunction.elu(x)
        else:
            raise ValueError('Invalid Activation Function')

    def forward(self, X):
        A = [X]

        # Compute activation for each hidden layer
        for i in range(len(self.weights)):
            Z = np.dot(A[i], self.weights[i]) + self.biases[i]
            A.append(self.activation_func(Z))

        return A[-1]

    def train(self, X, Y, learning_rate=0.01, epochs=100, optimizer='SGD', beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Define optimizer-specific variables
        if optimizer == 'Momentum':
            velocities = [np.zeros_like(w) for w in self.weights]
            gamma = 0.9
        elif optimizer == 'Adam':
            m = [np.zeros_like(w) for w in self.weights]
            v = [np.zeros_like(w) for w in self.weights]
        else:
            if optimizer != 'SGD':
                raise ValueError('Invalid Optimizer')

        for epoch in range(epochs):
            # Forward propagation
            A = [X]
            Z_list = []

            # Compute activation for each hidden layer
            for i in range(len(self.weights)):
                Z = np.dot(A[i], self.weights[i]) + self.biases[i]
                Z_list.append(Z)
                A.append(self.activation_func(Z))

            # Compute error for output layer
            error = Y - A[-1]
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}: loss = {loss}")

            dA = [error]

            # Backpropagation
            for i in range(len(self.weights) - 1, -1, -1):
                dZ = dA[-1] * (self.activation_func(Z_list[i]) * (1 - self.activation_func(Z_list[i])))
                dA.append(np.dot(dZ, self.weights[i].T))

                # Compute gradients using chosen optimizer
                if optimizer == 'SGD':
                    self.weights[i] += learning_rate * np.dot(A[i].T, dZ)
                    self.biases[i] += learning_rate * np.sum(dZ, axis=0)
                elif optimizer == 'Momentum':
                    velocities[i] = gamma * velocities[i] + learning_rate * np.dot(A[i].T, dZ)
                    self.weights[i] += velocities[i]
                    self.biases[i] += learning_rate * np.sum(dZ, axis=0)
                elif optimizer == 'Adam':
                    m[i] = beta1 * m[i] + (1 - beta1) * np.dot(A[i].T, dZ)
                    v[i] = beta2 * v[i] + (1 - beta2) * np.dot(A[i].T, dZ) ** 2
                    m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                    v_hat = v[i] / (1 - beta2 ** (epoch + 1))
                    self.weights[i] += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    self.biases[i] += learning_rate * np.sum(dZ, axis=0)

        return self.weights, self.biases

    def predict(self, X):
        return self.forward(X)