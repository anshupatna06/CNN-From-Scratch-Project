import numpy as np

class SoftmaxLoss:
    def forward(self, scores, y):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y = y

        N = scores.shape[0]
        loss = -np.sum(np.log(self.probs[np.arange(N), y])) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        dout = self.probs.copy()
        dout[np.arange(N), self.y] -= 1
        return dout / N
