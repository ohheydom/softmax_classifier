import numpy as np
from softmax import Softmax

sm = Softmax()
X = np.array([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4], [3,3,3,3]])
y = np.array([0, 1, 2, 3, 4, 3])
W = np.random.randn(X.shape[1], y.max() + 1)
b = np.zeros([y.max() + 1])
reg_strength = 1e-5
loss, dW = sm.loss(X, y, W, b, reg_strength)
print dW
