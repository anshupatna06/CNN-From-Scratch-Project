import numpy as np
from col2im import col2im

def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    N, F, out_h, out_w = dout.shape
    F, C, FH, FW = W.shape

    dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, F)

    # Gradient w.r.t. biases
    db = np.sum(dout_reshaped, axis=0)

    # Gradient w.r.t. weights
    dW = dout_reshaped.T.dot(X_col)
    dW = dW.reshape(W.shape)

    # Gradient w.r.t input (col2im)
    W_col = W.reshape(F, -1)
    dX_col = dout_reshaped.dot(W_col)

    dX = col2im(dX_col, X.shape, FH, FW, stride, padding)

    return dX, dW, db
