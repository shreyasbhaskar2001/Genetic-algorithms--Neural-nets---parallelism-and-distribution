import numpy as np
from numba import njit, prange

# Function to initialize neural network parameters
@njit
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Activation function (ReLU)
@njit
def relu(Z):
    return np.maximum(0, Z)

# Softmax function
@njit
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0))
    return expZ / np.sum(expZ, axis=0)

# Forward propagation
@njit
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Cross-entropy loss function
@njit
def cross_entropy_loss(Y, A2):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A2)) / m
    return loss

# Backward propagation
@njit
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

# Update parameters
@njit
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Main training function
@njit(parallel=True)
def train(X, Y, input_size, hidden_size, output_size, num_iterations, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    for i in prange(num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = cross_entropy_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")
    return W1, b1, W2, b2

if __name__ == "__main__":
    # Sample data for training
    np.random.seed(1)
    X = np.random.randn(3, 300)
    Y = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                  [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Neural network parameters
    input_size = X.shape[0]
    hidden_size = 5
    output_size = Y.shape[0]

    # Training parameters
    num_iterations = 1000
    learning_rate = 0.1

    # Train the neural network
    W1, b1, W2, b2 = train(X, Y, input_size, hidden_size, output_size, num_iterations, learning_rate)
