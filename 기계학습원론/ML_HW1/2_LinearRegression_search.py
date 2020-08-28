import numpy as np
from models.LinearRegression import LinearRegression

import matplotlib.pyplot as plt
from utils import optimizer, RMSE, load_data

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('Graduate')
x_train_data, y_train_data = train_data[0], train_data[1]
x_test_data, y_test_data = test_data[0], test_data[1]

# Hyper-parameter
_epoch=10000
_optim = 'SGD'

# ========================= EDIT HERE ========================
"""
Choose param to search. (batch_size or lr)
Specify values of the parameter to search,
and fix the other.
e.g.)
search_param = 'lr'
_batch_size = 32
_lr = [0.1, 0.01, 0.05]
"""
search_param = 'batch_size'
if search_param == 'lr':
    _batch_size = 32
    _lr = [0.001,0.005,0.01,0.05,0.1,0.2]
else:
    _batch_size = [4,8,16,32,64,128]
    _lr = 0.01
# ============================================================


train_results = []
test_results = []
search_space = _lr if search_param == 'lr' else _batch_size
for i, space in enumerate(search_space):
    # Build model
    model = LinearRegression(num_features=x_train_data.shape[1])
    optim = optimizer(_optim)

    # Train model with gradient descent
    if search_param == 'lr':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=space, optim=optim)
    else:
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=space, lr=_lr, optim=optim)
    
    ################### Evaluate on train data
    # Inference
    inference = model.eval(x_train_data)

    # Assess model
    error = RMSE(inference, y_train_data)
    print('[Search %d] RMSE on Train Data : %.4f' % (i+1, error))

    train_results.append(error)

    ################### Evaluate on test data
    # Inference
    inference = model.eval(x_test_data)

    # Assess model
    error = RMSE(inference, y_test_data)
    print('[Search %d] RMSE on test data : %.4f' % (i+1, error))

    test_results.append(error)

# ========================= EDIT HERE ========================
"""
Draw scatter plot of search results.
- X-axis: search paramter
- Y-axis: RMSE (Train, Test respectively)

Put title, X-axis name, Y-axis name in your plot.

Resources
------------
Official document: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html
"Data Visualization in Python": https://medium.com/python-pandemonium/data-visualization-in-python-scatter-plots-in-matplotlib-da90ac4c99f9
"""
if search_param == 'lr':
    plt.scatter(_lr,train_results)
    plt.scatter(_lr,test_results)
    plt.xlabel('learning rates')
    plt.ylabel('RMSE')
    plt.legend(['train','test'])
    plt.show()
else:
    plt.plot(_batch_size,train_results)
    plt.plot(_batch_size,test_results)
    plt.xlabel('batch sizes')
    plt.ylabel('RMSE')
    plt.legend(['train','test'])
    plt.show()
# ============================================================