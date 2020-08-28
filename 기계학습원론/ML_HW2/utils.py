import os
import numpy as np
from collections import Counter
import _pickle as pickle
import matplotlib.pyplot as plt
from model.SoftmaxClassifier import SoftmaxClassifier
from optim.Optimizer import *

def load_class_data(path, filename, target_at_front, to_binary=False, normalize=False, exclude_label=None, exclude_feature=None, shuffle=False):
    if exclude_feature is None:
        exclude_feature = []
    if exclude_label is None:
        exclude_label = []

    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    raw_data = lines[1:]
    num_feat = len(raw_data[0])
    feat_to_idx = [{} for _ in range(num_feat)]
    data = []
    for d in raw_data:
        line = []

        for i, f in enumerate(d):
            if i in exclude_feature:
                continue
            try:
                line.append(float(f))
            except:
                if f in feat_to_idx[i]:
                    f_idx = feat_to_idx[i][f]
                else:
                    f_idx = len(feat_to_idx[i])
                    feat_to_idx[i][f] = f_idx
                line.append(f_idx)
        data.append(line)

    data = np.array(data, dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0].astype(np.int32)
    else:
        x, y = data[:, :-1], data[:, -1].astype(np.int32)

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)

    if to_binary:
        y[y > 1] = 1

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y

def FashionMNISTData(path):
    train_x = np.load(os.path.join(path, 'train_images_full.npy'))
    train_y = np.load(os.path.join(path, 'train_labels_full.npy'))
    test_x = np.load(os.path.join(path, 'test_images_full.npy'))
    test_y = np.load(os.path.join(path, 'test_labels_full.npy'))

    bias = np.ones((train_x.shape[0], 1), dtype=np.float32)
    train_x = np.concatenate((bias, train_x), axis=1)

    bias = np.ones((test_x.shape[0], 1), dtype=np.float32)
    test_x = np.concatenate((bias, test_x), axis=1)
    
    return train_x, train_y, test_x, test_y

def IrisData(path, filename):
    x, y = load_class_data(path, filename, target_at_front=False, normalize=False, shuffle=True)
    return (x, y)

def accuracy(h, y):
    """
    h : (N, ), predicted label
    y : (N, ), correct label
    """
    # if len(h.shape) == 1:
    # h = np.expand_dims(h, 1)
    # if len(y.shape) == 1:
    # y = np.expand_dims(y, 1)

    total = y.shape[0]
    correct = len(np.where(h==y)[0])
    acc = correct / total

    return acc

data_dir = {
    'Fashion_mnist': 'fashion_mnist',
    'Iris': 'iris',
}

def load_data(data_name):
    dir_name = data_dir[data_name]
    path = os.path.join('./data', dir_name)

    if data_name == 'Iris':
        train_x, train_y = IrisData(path, 'train.csv')
        test_x, test_y = IrisData(path, 'test.csv')
    elif data_name == 'Fashion_mnist':
        train_x, train_y, test_x, test_y = FashionMNISTData(path)
    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)

def load_label_dict(dataset):
    if dataset == 'digit':
        return {i: str(i) for i in range(10)}
    elif dataset == 'Fashion_mnist':
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        return {i: labels[i] for i in range(10)}
    else:
        raise ValueError('Select correct dataset.')

def display_image_predictions(image, label, prob, label_dict, top_k=3):
    n_predictions = len(image)

    top_k_pred = np.argpartition(-prob, top_k, axis=1)[:, :top_k]
    top_k_prob = np.take_along_axis(prob, top_k_pred, 1)
    inner_sort_idx = np.argsort(-top_k_prob, axis=1)

    top_k_pred = np.take_along_axis(top_k_pred, inner_sort_idx, 1)
    top_k_prob = np.take_along_axis(top_k_prob, inner_sort_idx, 1)

    fig, axies = plt.subplots(nrows=n_predictions, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    margin = 0.05
    ind = np.arange(top_k)
    width = (1. - 2. * margin) / n_predictions

    for col, (img, label_id, pred_k, prob_k) in enumerate(zip(image, label, top_k_pred, top_k_prob)):
        axies[col][0].imshow(img)
        axies[col][0].set_title('Label: ' + label_dict[label_id])
        axies[col][0].set_axis_off()

        pred_k = [label_dict[p] for p in pred_k]

        axies[col][1].barh(ind + margin, prob_k[::-1], width)
        axies[col][1].set_yticks(ind + margin)
        axies[col][1].set_yticklabels(pred_k[::-1])
        axies[col][1].set_xticks([0, 0.5, 1.0])
    plt.show()
