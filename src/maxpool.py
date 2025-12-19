import numpy as np

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.cache = X
        N, C, H, W = X.shape
        PH = PW = self.pool_size
        stride = self.stride

        # ✅ robust output size
        out_h = (H - PH) // stride + 1
        out_w = (W - PW) // stride + 1

        out = np.zeros((N, C, out_h, out_w))

        for n in range(N):
            for c in range(C):
                out_i = 0
                for i in range(0, H - PH + 1, stride):
                    out_j = 0
                    for j in range(0, W - PW + 1, stride):
                        window = X[n, c, i:i+PH, j:j+PW]
                        out[n, c, out_i, out_j] = np.max(window)
                        out_j += 1
                    out_i += 1

        return out

    def backward(self, dout):
        X = self.cache
        N, C, H, W = X.shape
        PH = PW = self.pool_size
        stride = self.stride

        dX = np.zeros_like(X)

        out_h, out_w = dout.shape[2], dout.shape[3]

        for n in range(N):
            for c in range(C):
                out_i = 0
                for i in range(0, H - PH + 1, stride):
                    out_j = 0
                    for j in range(0, W - PW + 1, stride):
                        window = X[n, c, i:i+PH, j:j+PW]
                        mask = (window == np.max(window))
                        dX[n, c, i:i+PH, j:j+PW] += mask * dout[n, c, out_i, out_j]
                        out_j += 1
                    out_i += 1

        return dX
