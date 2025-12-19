import numpy as np

def im2col(X, FH, FW, stride=1, padding=0):
    N, C, H, W = X.shape
    out_h = (H + 2*padding - FH) // stride + 1
    out_w = (W + 2*padding - FW) // stride + 1

    X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

    col = np.zeros((N, C, FH, FW, out_h, out_w))

    for i in range(FH):
        i_max = i + stride * out_h
        for j in range(FW):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = X_padded[:, :, i:i_max:stride, j:j_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col
