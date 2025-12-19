import numpy as np

def col2im(col, X_shape, FH, FW, stride=1, padding=0):
    N, C, H, W = X_shape
    out_h = (H + 2*padding - FH) // stride + 1
    out_w = (W + 2*padding - FW) // stride + 1

    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0,3,4,5,1,2)

    X_padded = np.zeros((N, C, H + 2*padding, W + 2*padding))

    for i in range(FH):
        i_max = i + stride * out_h
        for j in range(FW):
            j_max = j + stride * out_w
            X_padded[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]

    return X_padded[:, :, padding:H+padding, padding:W+padding]
