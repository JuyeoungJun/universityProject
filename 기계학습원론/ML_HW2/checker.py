import random
import numpy as np
from optim.Optimizer import SGD
from model.SoftmaxClassifier import SoftmaxClassifier
from utils import accuracy

np.random.seed(428)

num_epochs = 10
batch_size = 3
learning_rate = 2

x = np.array([[1, 2, 1, 1],
            [2, 1, 3, 2], 
            [3, 1, 3, 4], 
            [4, 1, 5, 5], 
            [1, 7, 5, 5], 
            [1, 2, 5, 6], 
            [1, 6, 6, 6], 
            [1, 7, 7, 7]])
y = np.array([2, 2, 2, 1, 1, 1, 0, 0])

num_data, num_features = x.shape
num_label = int(y.max()) + 1
print('# of Training data : %d \n' % num_data)


model = SoftmaxClassifier(num_features, num_label)

# ================================== Softmax ==================================
"""
Correct output:
[[2.06106005e-09 4.53978686e-05 9.99954600e-01]
 [3.33333333e-01 3.33333333e-01 3.33333333e-01]
 [9.00305732e-02 2.44728471e-01 6.65240956e-01]
 [1.94615163e-03 3.90895004e-02 9.58964348e-01]]
"""
print('1. check "_softmax"')
softmax_in = np.array([
    [-10, 0, 10],
    [0, 0, 0],
    [100, 101, 102],
    [-0.7, 2.3, 5.5]])
softmax_out = model._softmax(softmax_in)
print('Softmax out:')
print(softmax_out)

# ==================================  Train  ==================================
"""
Correct output:
[[-24.40371188  12.30451339  12.09919851]
 [ 24.01838466 -24.48808664   0.46970179]
 [  9.83380932  -4.4597069   -5.37410031]
 [  6.56554885   1.54029524  -8.10584245]]
"""
optim = SGD()
model.train(x, y, num_epochs, batch_size, learning_rate, optim)
print('Trained weight:')
print(model.W)
print()

# ==================================  Eval   ==================================
"""
Correct output:
Accuracy on train data : 0.375000
"""
pred, prob = model.eval(x)

train_acc = accuracy(pred, y)
print('Accuracy on train data : %f\n' % train_acc)
