import random
import numpy as np
from optim.Optimizer import SGD
from model.SoftmaxClassifier import SoftmaxClassifier
from utils import load_data, accuracy, display_image_predictions, load_label_dict

np.random.seed(428)

# ========================= EDIT HERE =========================
# 1. Choose DATA : Fashion_mnist, Iris
# 2. Adjust Hyperparameters

# DATA
DATA_NAME = 'Iris'

# HYPERPARAMETERS
num_epochs = [1000]
batch_size = 10
learning_rate = [0.001, 0.000001]

show_plot = False
# =============================================================
assert DATA_NAME in ['Iris', 'Fashion_mnist']
grid_search = [(x, y) for x in num_epochs for y in learning_rate]

# Load dataset, model and evaluation metric
train_data, test_data = load_data(DATA_NAME)
train_x, train_y = train_data

num_train = len(train_x)
perm = np.random.permutation(num_train)
num_valid = int(len(train_x) * 0.1)
valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
train_x, train_y = train_x[train_idx], train_y[train_idx]

num_data, num_features = train_x.shape
num_label = int(train_y.max()) + 1
print('# of Training data : %d \n' % num_data)

results = {}
best_acc = -1
best_model = None

# For each set of parameters in 'grid_search', train and evaluate softmax classifier.
# Save search history in dictionary 'results'.
#   - KEY: tuple of (# of epochs, learning rate)
#   - VALUE: accuracy on validation data
# Save the best validation accuracy and optimized model in 'best_acc' and 'best_model'.

for ep, lr in grid_search:
    # Make model & optimizer
    model = SoftmaxClassifier(num_features, num_label)
    optim = SGD()

    model.train(train_x, train_y, ep, batch_size, lr, optim)

    pred, prob = model.eval(valid_x)

    valid_acc = accuracy(pred, valid_y)
    print('Accuracy on valid data : %f\n' % valid_acc)

    results[ep, lr] = valid_acc
    
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = model

for ep, lr in sorted(results):
    valid_acc = results[(ep, lr)]
    print('# epochs : %d lr : %e valid accuracy : %f' % (ep, lr, valid_acc))
    
print('best validation accuracy achieved: %f' % best_acc)

# Evaluate best model on test data
test_x, test_y = test_data
pred, prob = best_model.eval(test_x)

test_acc = accuracy(pred, test_y)
print('test accuracy of best model : %f' % test_acc)

# Plot prediction of the best model
if show_plot and DATA_NAME == 'Fashion_mnist':
    num_test = len(test_x)
    test_x = test_x[:, 1:]
    test_x = test_x.reshape(num_test, 28, 28)
    
    random_idx = np.random.choice(num_test, 5)
    sample_data = test_x[random_idx]
    sample_label = test_y[random_idx]
    sample_prob = prob[random_idx]

    label_dict = load_label_dict(DATA_NAME)
    
    display_image_predictions(sample_data, sample_label, sample_prob, label_dict)
