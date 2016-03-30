# Softmax Classifier

This softmax classifier uses training data with labels to build a model which can then predict labels on other samples. It implements multiple weight update algorithms including Adam, RMSProp, SGD, and SGD with Momentum.

## Usage

Load your training data and split into two variables, features and labels.

```python
import numpy as np
from softmax import Softmax

X = your_data
y = your_data_labels
reg_strength = 1e-4
batch_size = 50
epochs = 1000
learning_rate = 1
weight_update = 'sgd_with_momentum'
clf = Softmax(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, reg_strength=reg_strength, weight_update=weight_update)
clf.train(X, y)
pred = clf.predict(X)
print np.mean(np.equal(y, pred))
```

See `sample.py` for a full example.

### Weight update parameters

You can use different algorithms for the weight updates:

* adam
* sgd_with_momentum
* sgd
* rms_prop
