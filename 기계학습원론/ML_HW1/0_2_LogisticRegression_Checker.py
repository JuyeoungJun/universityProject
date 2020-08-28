import numpy as np
from models.LogisticRegression import LogisticRegression

from utils import optimizer, Accuracy

np.random.seed(10)

Dataset = np.loadtxt('data/logistic_check_data.txt')
x_data, y_data = Dataset[:, :-1], Dataset[:, -1]

_epoch = 100
_batch_size = 5
_lr = 0.01
_optim = 'SGD'

#======================================================================================================
print('='*20, 'Sigmoid Test', '='*20)
test_case_1 = np.array([0.5, 0.5, 0.5])
test_case_2 = np.array([
    [6.23, -7.234, 8.3],
    [-1, -6.23, -9]
])
test_case_3 = np.array([
    [[1.0, 1.1], [5.672, -4]],
    [[0.0, 9], [-9, 0.1]]
])
test_result_1 = LogisticRegression._sigmoid(None, test_case_1)
test_result_2 = LogisticRegression._sigmoid(None, test_case_2)
test_result_3 = LogisticRegression._sigmoid(None, test_case_3)

print('## Test case 1')
print('Input:\n', test_case_1)
print('Output:\n', test_result_1, end='\n\n')
print('## Test case 2')
print('Input:\n', test_case_2)
print('Output:\n', test_result_2, end='\n\n')
print('## Test case 3')
print('Input:\n', test_case_3)
print('Output:\n', test_result_3, end='\n\n')
'''
You should get results as:

## Test case 1
Input:
 [0.5 0.5 0.5]
Output:
 [0.62245933 0.62245933 0.62245933]

## Test case 2
Input:
 [[ 6.23  -7.234  8.3  ]
 [-1.    -6.23  -9.   ]]
Output:
 [[9.98034419e-01 7.21108196e-04 9.99751545e-01]
 [2.68941421e-01 1.96558078e-03 1.23394576e-04]]

## Test case 3
Input:
 [[[ 1.     1.1  ]
  [ 5.672 -4.   ]]

 [[ 0.     9.   ]
  [-9.     0.1  ]]]
Output:
 [[[7.31058579e-01 7.50260106e-01]
  [9.96570823e-01 1.79862100e-02]]

 [[5.00000000e-01 9.99876605e-01]
  [1.23394576e-04 5.24979187e-01]]]

'''

#======================================================================================================

print('='*20, 'Logistic Regression Test', '='*20)

model = LogisticRegression(num_features=x_data.shape[1])
optimizer = optimizer(_optim)
print('Initial weight: \n', model.W.reshape(-1))
print()

model.fit(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer)
print('Trained weight: \n', model.W.reshape(-1))
print()


# model evaluation
inference = model.eval(x_data)

# Error calculation
error = Accuracy(inference, y_data)
print('Accuracy on Check Data : %.4f \n' % error)

'''
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight: 
 [-0.30839267  0.07120854  0.27459075  0.08573039  0.34718609]

Accuracy on Check Data : 0.8000

'''
