import os
import numpy as np

def load_mnist(data_path):
    mnist_path = os.path.join(data_path, 'mnist')

    x_train = np.load(os.path.join(mnist_path, 'mnist_train_x.npy'))
    y_train = np.load(os.path.join(mnist_path, 'mnist_train_y.npy'))
    x_test = np.load(os.path.join(mnist_path, 'mnist_test_x.npy'))
    y_test = np.load(os.path.join(mnist_path, 'mnist_test_y.npy'))

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    # Y as one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def rel_error(x, y):
    # return np.sum(np.abs(x-y))
    diff_abs = np.abs(x-y)
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    return np.max(diff_abs / (np.maximum(1e-8, x_abs + y_abs)))

def check_conv_validity(x, w, stride, pad):
    N, C, W, H = x.shape
    F, _, WW, HH = w.shape
    
    # ensure square kernel
    assert HH == WW

    # # in_channel of input and kernel should match.
    if x.shape[1] != w.shape[1]:
        raise ValueError('Input and kernel # channel mismatch.')
    
    # input width, height >= kernel
    if (W + 2*pad) < WW or (H + 2*pad) < WW:
        raise ValueError('Kernel size is larger than input size')
    # kernel size & stride mismatch with input
    if (W + 2*pad - WW) % stride != 0 or (H + 2*pad - WW) % stride != 0:
        remain = (W + 2*pad - WW) % stride if (W + 2*pad - WW) % stride != 0 else (H + 2*pad - WW) % stride
        raise ValueError(f'Input size {W}, Kernel size {WW}, stride {stride}, pad {pad} mismatch: {remain} row(s) or column(s) remain.')

def check_pool_validity(x, HH, stride):
    N, C, W, H = x.shape
    
    # input width, height >= kernel
    if W < HH or H < HH:
        raise ValueError('Pooling size is larger than input size')
    # kernel size & stride mismatch with input
    if (W - HH) % stride != 0 or (H - HH) % stride != 0:
        remain = (W - HH) % stride if (W - HH) % stride != 0 else (H - HH) % stride
        raise ValueError(f'Input size {W}, Pooling size {HH}, stride {stride} mismatch: {remain} row(s) or column(s) remain.')
