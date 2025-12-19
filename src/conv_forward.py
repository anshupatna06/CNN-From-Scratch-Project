import numpy as np
from im2col import im2col

def conv_forward(X, W, b, stride=1, padding=0):
    N, C, H, W_input = X.shape
    F, _, FH, FW = W.shape

    X_col = im2col(X, FH, FW, stride, padding)
    W_col = W.reshape(F, -1)

    out = X_col.dot(W_col.T) + b
    out_h = (H + 2*padding - FH) // stride + 1
    out_w = (W_input + 2*padding - FW) // stride + 1

    out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    cache = (X, W, b, stride, padding, X_col)
    return out, cache
