import numpy as np

class ReLU:
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        X = self.cache
        dX = dout.copy()
        dX[X <= 0] = 0
        return dX
