import numpy as np
from models.LogisticRegression import LogisticRegression

import matplotlib.pyplot as plt
from utils import load_data, optimizer, Accuracy

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('RedWine')
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

# Hyper-parameter
_epoch=1000
_batch_size=32
_lr = 0.001
_optim = 'SGD'

# Build model
model = LogisticRegression(num_features=x_train.shape[1])
optimizer = optimizer(_optim)

# Solve
print('Train start!')
model.fit(x=x_train, y=y_train, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer)
print('Trained done.')

# Inference
print('Predict on test data')
inference = model.eval(x_test)

# Assess model
error = Accuracy(inference, y_test)
print('Accuracy on Test Data : %.4f' % error)