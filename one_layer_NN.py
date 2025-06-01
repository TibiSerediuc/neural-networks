import numpy as np 

# I know that sigmoid is not really used anymore, but I use it for learning purposes
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# it ill be used for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array( [[0, 0], [0, 1], [1, 0], [1, 1]] )
y = np.array( [[0], [0], [0], [1]] )

# we randomly initialise the weights and biases
np.random.seed(42)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):

    # Forward pass
    z = np.dot(X, weights) + bias
    a = sigmoid(z)

    loss = np.mean((a - y) ** 2)

    dz = (a - y) * sigmoid_derivative(a)
    dw = np.dot(X.T, dz) / X.shape[0]
    db = np.mean(dz)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 1000 == 0:
        print(f"Epoxh {epoch}, Loss: {loss:.4f}")

#Final predictions
print(sigmoid(np.dot(X, weights) + bias))
