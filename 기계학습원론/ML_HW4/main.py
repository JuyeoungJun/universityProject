import time
import numpy as np
from utils import load_mnist
from Answer import CNN_Classifier, ConvolutionLayer, MaxPoolingLayer, FCLayer, SoftmaxLayer, ReLU, Sigmoid, Tanh
import matplotlib.pyplot as plt
import copy

np.random.seed(2020)
np.tanh = lambda x: x

CNN = CNN_Classifier()

# =============================== EDIT HERE ===============================
################################################################################################################################
# Build model Architecture and do experiment.
# You can add layer as below examples.
# NOTE THAT, layers are executed in the order that you add.
# 
# < Add Layers examples >
# - Convolution Layer
#     CNN.add_layer('Conv Example Layer', ConvolutionLayer(in_channels=in_channnel, out_channels=16, kernel_size=5, pad=1))
# - FC Layer
#     model.add_layer('FC Example Layer', FCLayer(input_dim=1234, output_dim=123))
# - Softmax Layer
#     model.add_layer('Softmax Layer', SoftmaxLayer())
################################################################################################################################

# Hyper-parameters
num_epochs = 10
learning_rate = 0.01
reg_lambda = 0.001
print_every = 1

batch_size = 1000

# =========================================================================

x_train, y_train, x_test, y_test = load_mnist('./data')

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, in_channels, H, W = x_train.shape
num_class = y_train.shape[1]

train_accuracy = []
valid_accuracy = []

best_epoch = -1
best_acc = -1
best_model = None

# =============================== EDIT HERE ===============================

# Add layers
CNN.add_layer('Conv-1', ConvolutionLayer(in_channels=in_channels, out_channels=4, kernel_size=3, pad=1))
CNN.add_layer('ReLU-1',ReLU())
CNN.add_layer('Conv-2', ConvolutionLayer(in_channels=4, out_channels=4, kernel_size=3, pad=1))
CNN.add_layer('ReLU-2',ReLU())
CNN.add_layer('Max-pool-1',MaxPoolingLayer(2,2))
CNN.add_layer('FC-1',FCLayer(784,500))
CNN.add_layer('ReLU-3',ReLU())
CNN.add_layer('FC-2',FCLayer(500,10))
CNN.add_layer('Softmax Layer',SoftmaxLayer())

# =========================================================================

CNN.summary()

print('Training Starts...')
num_batch = int(np.ceil(num_train / batch_size))

for epoch in range(1, num_epochs + 1):
    start = time.time()
    epoch_loss = 0.0
    for b in range(num_batch):
        s = b * batch_size
        e = (b+1) * batch_size if (b+1) * batch_size < len(x_train) else len(x_train)

        x_batch = x_train[s:e]
        y_batch = y_train[s:e]

        loss = CNN.forward(x_batch, y_batch, reg_lambda)
        epoch_loss += loss

        CNN.backward(reg_lambda)
        CNN.update(learning_rate)

        print('[%4d / %4d]\t batch loss : %.4f' % (s, num_train, loss))

    end = time.time()
    diff = end - start
    print('Epoch %d took %.2f seconds\n' % (epoch, diff))

    if epoch % print_every == 0:
        # TRAIN ACCURACY
        prob = CNN.predict(x_train)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_train, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        train_acc = correct / total
        train_accuracy.append(train_acc)

        # VAL ACCURACY
        prob = CNN.predict(x_valid)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_valid, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        valid_acc = correct / total
        valid_accuracy.append(valid_acc)

        print('EPOCH %d Loss = %.5f' % (epoch, epoch_loss))
        print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)

        if best_acc < valid_acc:
            print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
            best_acc = valid_acc
            best_epoch = epoch
            best_model = copy.deepcopy(CNN)
        print()

print('Training Finished...!!')
print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

# TEST ACCURACY
prob = best_model.predict(x_test)
pred = np.argmax(prob, -1).astype(int)
true = np.argmax(y_test, -1).astype(int)

correct = len(np.where(pred == true)[0])
total = len(true)
test_acc = correct / total

print('Test Accuracy at Best Epoch : %.2f' % (test_acc))


# =============================== EDIT HERE ===============================
##################################################################                                                               
# Draw a plot of train/valid accuracy.
# X-axis : Epoch
# Y-axis : train_accuracy & valid_accuracy
# Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
##################################################################

epochs = list(np.arange(1, num_epochs+1, print_every))

plt.plot(epochs, train_accuracy, label='Train Acc.')
plt.plot(epochs, valid_accuracy, label='Valid Acc.')

plt.title('Epoch - Train/Valid Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# =========================================================================
plt.show()
