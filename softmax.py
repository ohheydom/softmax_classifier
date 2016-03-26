import numpy as np
import math
import random

class Softmax:
    def __init__(self, batch_size=50, epochs=100, learning_rate=1e-3, reg_strength=1e-5):
        self.W = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength

    def train(self, X, y):
        self.W = np.random.randn(X.shape[1], y.max() + 1)
        config = {}
        for epoch in range(self.epochs):
            #loss = self.sgd(X, y, self.learning_rate, self.batch_size, self.reg_strength)
            config, loss = self.sgd_with_momentum(X, y, self.learning_rate, self.batch_size, self.reg_strength, config)
            print "Epoch: %s, Loss: %s" % (epoch, loss)

    def predict(self, X):
        return np.argmax(X.dot(self.W), 1)

    def loss(self, X, y, W, b, reg_strength):
        sample_size = X.shape[0]
        predictions = X.dot(W) + b

        # Fix numerical instability
        predictions -= predictions.max(axis=1).reshape([-1, 1])

        # Run predictions through softmax
        softmax = math.e**predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])

        # Cross entropy loss
        loss = -np.log(softmax[np.arange(len(softmax)), y]).sum() / sample_size
        loss += 0.5*reg_strength * (W*W).sum()

        softmax[np.arange(len(softmax)), y] -= 1
        dW = X.T.dot(softmax) / sample_size
        dW += reg_strength * W
        return loss, dW

    def sgd(self, X, y, learning_rate, batch_size, reg_strength):
        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        loss, dW = self.loss(X_batch, y_batch, self.W, 0, reg_strength)
        self.W -= learning_rate * dW
        return loss

    def sgd_with_momentum(self, X, y, learning_rate, batch_size, reg_strength, config=None):
        v = config.get('velocity', np.zeros(self.W.shape))
        momentum = config.get('momentum', 0.9)

        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        loss, dW = self.loss(X_batch, y_batch, self.W, 0, reg_strength)

        v = momentum*v - learning_rate*dW
        self.W += v
        config['velocity'] = v
        config['momentum'] = momentum
        return config, loss
