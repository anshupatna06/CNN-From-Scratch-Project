import numpy as np
from conv_forward import conv_forward
from conv_backward import conv_backward
from relu import ReLU
from maxpool import MaxPool
from flatten import Flatten
from dense import Dense
from softmax_loss import SoftmaxLoss


class SimpleCNN:
    def __init__(self):
        # Layers
        self.relu = ReLU()
        self.pool = MaxPool(pool_size=2, stride=2)
        self.flatten = Flatten()
        self.loss_fn = SoftmaxLoss()

        # Convolution parameters
        self.num_filters = 8
        self.filter_size = 3

        self.W = 0.01 * np.random.randn(
            self.num_filters, 1, self.filter_size, self.filter_size
        )
        self.b = np.zeros(self.num_filters)

        # 🔥 CORRECT Dense input size
        # After conv (padding=1): 28x28
        # After pool (2x2): 14x14
        # Channels = num_filters
        fc_input_dim = self.num_filters * 14 * 14

        self.fc = Dense(fc_input_dim, 10)

    def forward(self, X, y=None):
        # Conv → ReLU → Pool
        out, self.conv_cache = conv_forward(
            X, self.W, self.b, stride=1, padding=1
        )
        out = self.relu.forward(out)
        out = self.pool.forward(out)

        # Flatten → FC
        out = self.flatten.forward(out)
        scores = self.fc.forward(out)

        if y is None:
            return scores

        loss = self.loss_fn.forward(scores, y)
        return loss

    def backward(self):
        # Loss → FC → Flatten → Pool → ReLU → Conv
        dout = self.loss_fn.backward()
        dout = self.fc.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.pool.backward(dout)
        dout = self.relu.backward(dout)

        dX, dW, db = conv_backward(dout, self.conv_cache)

        # Update conv parameters
        self.W -= 0.01 * dW
        self.b -= 0.01 * db
