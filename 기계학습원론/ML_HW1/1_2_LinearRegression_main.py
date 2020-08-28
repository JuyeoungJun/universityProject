import numpy as np
from models.LinearRegression import LinearRegression

import matplotlib.pyplot as plt
from utils import optimizer, RMSE, load_data

np.random.seed(2020)

# Data generation
train_data, _ = load_data('Concrete')
x_data, y_data = train_data[0], train_data[1]

# Hyper-parameter
_epoch=10000
_batch_size=32
_lr = 0.01
_optim = 'SGD'

# ========================= EDIT HERE ========================
'''
Change GD to :
True for numerical solution
False for analytic solution
'''

GD = True
# ============================================================

# Build model
model = LinearRegression(num_features=x_data.shape[1])
optimizer = optimizer(_optim)
print('Initial weight: \n', model.W.reshape(-1))

# Solve
if GD:
    model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer)
else:
    model.analytic_solution(x=x_data, y=y_data)

print('Trained weight: \n', model.W.reshape(-1))

# Inference
inference = model.eval(x_data)

# Assess model
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f' % error)
