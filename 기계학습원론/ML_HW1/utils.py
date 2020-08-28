import numpy as np
import os
from models.LinearRegression import LinearRegression
from optim.Optimizer import *


def RMSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse

def Accuracy(h, y):
    total = h.shape[0]
    correct = len(np.where(h==y)[0])
    accuracy = correct / total

    return accuracy

def optimizer(optim_name):
    if optim_name == 'SGD':
        optim = SGD()
    else:
        raise NotImplementedError
    return optim


def load_data(data_name):
    path = os.path.join('./data', data_name)

    if data_name == 'Concrete':
        train_x, train_y = ConcreteData(path, 'train.csv')
        test_x, test_y = ConcreteData(path, 'test.csv')
    elif data_name == 'Graduate':
        train_x, train_y = GraduateData(path, 'train.csv')
        test_x, test_y = GraduateData(path, 'test.csv')
    elif data_name == 'Titanic':
        train_x, train_y = TitanicData(path, 'train.csv')
        test_x, test_y = TitanicData(path, 'test.csv')
    elif data_name == 'RedWine':
        train_x, train_y = RedWineData(path, 'train.csv')
        test_x, test_y = RedWineData(path, 'test.csv')
    else:
        raise ValueError('%d dataset is not supported.' % data_name)

    return (train_x, train_y), (test_x, test_y)


def ConcreteData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)

def GraduateData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)

def TitanicData(path, filename):
    return load_class_data(path, filename, target_at_front=True, normalize=False, exclude=[2])

def RedWineData(path, filename):
    return load_class_data(path, filename, target_at_front=False, normalize=False)

def load_reg_data(path, filename, target_at_front, normalize=False, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((x, bias), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y

def load_class_data(path, filename, target_at_front, normalize=False, exclude=[]):
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
            if i in exclude:
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
    x = np.concatenate((x, bias), axis=1)

    return x, y