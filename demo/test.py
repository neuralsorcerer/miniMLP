import sys
sys.path.append('../')

import numpy as np
import pickle

# Load the MLP model from a file
with open('mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Use the trained model to make predictions on new data
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = model.predict(X_test)

# Print the predictions
print(y_pred)
