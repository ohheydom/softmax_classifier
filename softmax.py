import numpy as np
import math

class Softmax:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=0.001):
        pass


    def predict(self, X):
        pass

    def loss(self, X, y, W, b, reg_strength):
        sample_size = X.shape[0]
        predictions = X.dot(W) + b

        # Fix numerical instability
        predictions -= predictions.max(axis=1).reshape([-1, 1])
        softmax = math.e**predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        # Cross entropy loss
        loss = -np.log(softmax[np.arange(len(softmax)), y]).sum() / sample_size
        loss += 0.5*reg_strength * (W*W).sum()

        softmax[np.arange(len(softmax)), y] -= 1
        dW = X.T.dot(softmax) / sample_size
        dW += reg_strength * W
        return loss, dW
