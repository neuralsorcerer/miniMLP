import sys
sys.path.append('../')

import numpy as np
from miniMLP.engine import MLP
import pickle

# Define the training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Initialize the MLP
mlp = MLP(input_size=2, output_size=1, hidden_layers=3, hidden_sizes=[4, 6, 4], activation='relu')

# Train the MLP using the training data
mlp.train(X_train, y_train, epochs=10000, learning_rate=0.0001)

# Save the MLP model to a file
with open('mlp_model.pkl', 'wb') as file:
    pickle.dump(mlp, file)
