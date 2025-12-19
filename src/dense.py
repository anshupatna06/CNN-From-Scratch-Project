import numpy as np

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = 0.01 * np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        dX = dout @ self.W.T
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)

        self.W -= 0.01 * dW
        self.b -= 0.01 * db

        return dX
