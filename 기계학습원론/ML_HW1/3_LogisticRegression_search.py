import numpy as np
from models.LogisticRegression import LogisticRegression

import matplotlib.pyplot as plt
from utils import load_data, optimizer, Accuracy

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('Titanic')
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

# ========================= EDIT HERE ========================
'''
Data feature engineering.
Extract features from raw data, if you want, as you wish.

Description of each column in 'x_train' & 'x_test' is specified as follows:
    - Column 0 Pclass: Ticket class, Categorical, (1st: 0, 2nd: 1, 3rd: 2)
    - Column 1 Sex: Sex, Categorical, {male: 0, female: 1}
    - Column 2 Age: Age. Numeric, float.
    - Column 3 Siblings/Spouses Aboard: # of siblings/spouses aboard with, Numeric, integer.
    - Column 4 Parents/Children Aboard: # of parents/children aboard with, Numeric, integer.
    - Column 5 Fare: Fare of a passenger, Numeric, float.
    - Column 6 Bias: Bias initialized with 1.
'''
def feature_func_(x):
    # AS YOU WISH
    # DEFAULT: DO NOTHING.
    """
    for k in x:
        if(k[2]<10):
            k[2] = 0
        elif(k[2] >= 15 and k[2] < 30):
            k[2] = 1
        elif(k[2] >= 30 and k[2] < 45):
            k[2] = 2
        elif(k[2] >= 45 and k[2] < 60):
            k[2] = 3
        else:
            k[2] = 4
        k[3] += k[4]
        if(k[5] < 20):
            k[5] = 0
        elif(k[5] >= 20 and k[5] < 40):
            k[5] = 1
        elif(k[5] >= 40 and k[5] < 100):
            k[5] = 2
        else:
            k[5] = 3
    x = np.delete(x,4,axis=1)
    x = np.delete(x,4,axis=1)
    """
    x = np.delete(x,5,axis=1)
    x = (x-np.mean(x))/np.std(x)
    return x

# ============================================================
x_new_data = feature_func_(x_train)
assert len(x_train) == len(x_new_data), '# of data must be same.'

# Hyper-parameter
_optim = 'SGD'
_batch_size=50
# ========================= EDIT HERE ========================
'''
Tuning hyper-parameters.
Here, tune two kinds of hyper-parameters, 
# of epochs (_epoch) and learning_rate (_lr).

'''
_epoch=1000000
_lr = 0.0015
# ============================================================

# Build model
model = LogisticRegression(num_features=x_new_data.shape[1])
optimizer = optimizer(_optim)

# Solve
print('Train start.')
model.fit(x=x_new_data, y=y_train, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer)
print('Trained done.')

# Inference
print('Predict on test data')
inference = model.eval(feature_func_(x_test))

# Assess model
error = Accuracy(inference, y_test)
print('Accuracy on test data : %.4f' % error)
