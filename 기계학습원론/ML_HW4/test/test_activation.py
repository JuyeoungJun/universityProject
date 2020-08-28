import unittest
import numpy as np
from Answer import ReLU, Sigmoid, Tanh
from utils import rel_error

class TestSigmoid(unittest.TestCase):
    def test_sigmoid_1_forward(self):
        print('\n==================================')
        print('        Test sigmoid forward      ')
        print('==================================')
        x = np.linspace(-1.5, 1.5, num=30).reshape(6, 5)
        sigmoid = Sigmoid()
        out = sigmoid.forward(x)
        correct_out = [[0.18242552, 0.19836387, 0.21532798, 0.23332061, 0.25233322],
                       [0.27234476, 0.29332071, 0.31521239, 0.33795656, 0.36147556],
                       [0.3856778 , 0.41045882, 0.4357028 , 0.46128455, 0.48707185],
                       [0.51292815, 0.53871545, 0.5642972 , 0.58954118, 0.6143222 ],
                       [0.63852444, 0.66204344, 0.68478761, 0.70667929, 0.72765524],
                       [0.74766678, 0.76667939, 0.78467202, 0.80163613, 0.81757448]]

        e = rel_error(correct_out, out)
        print('Relative difference:', e)

        self.assertTrue(e <= 5e-8)

    def test_sigmoid_2_backward(self):
        print('\n==================================')
        print('        Test sigmoid backward     ')
        print('==================================')
        np.random.seed(123)
        x = np.random.randn(5, 5)
        sigmoid = Sigmoid()
        out = sigmoid.forward(x)
        dx = sigmoid.backward(x, 0.0)
        correct_dx = [[-0.2048748 ,  0.19633044,  0.06934706, -0.22376078, -0.13318844],
                      [ 0.22297909, -0.18096644, -0.10244393,  0.21720183, -0.18059114],
                      [-0.15157559, -0.02362423,  0.2236528 , -0.14447372, -0.10570044],
                      [-0.10362309,  0.1971545 ,  0.1984592 ,  0.19703887,  0.09303448],
                      [ 0.16139223,  0.22364718, -0.18937883,  0.21188431, -0.21658554]]
        e = rel_error(correct_dx, dx)
        print('Relative difference:', e)
        self.assertTrue(e <= 1e-7)
    
    def runTest(self):
        self.test_sigmoid_1_forward()
        self.test_sigmoid_2_backward()

class TestReLU(unittest.TestCase):
    def test_relu_1_forward(self):
        print('\n==================================')
        print('          Test ReLU forward       ')
        print('==================================')
        x = np.linspace(-0.7, 0.5, num=20).reshape(5, 4)
        relu = ReLU()
        out = relu.forward(x)
        correct_out = np.array([[0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0.05789474, 0.12105263, 0.18421053, 0.24736842],
                                [0.31052632, 0.37368421, 0.43684211, 0.5]])
        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-08)

    def test_relu_2_backward(self):
        print('\n==================================')
        print('          Test ReLU backward      ')
        print('==================================')
        np.random.seed(123)
        relu = ReLU()
        x = np.random.randn(7, 7)
        d_prev = np.random.randn(*x.shape)
        out = relu.forward(x)
        dx = relu.backward(d_prev, 0.0)
        correct_dx = [[0.        , -1.29408532, -1.03878821,  0.        ,  0.        ,  0.02968323, 0.        ],
                      [0.        ,  1.75488618,  0.        ,  0.        ,  0.        ,  0.79486267, 0.        ],
                      [0.        ,  0.        ,  0.80723653,  0.04549008, -0.23309206, -1.19830114, 0.19952407],
                      [0.46843912,  0.        ,  1.16220405,  0.        ,  0.        ,  1.03972709, 0.        ],
                      [0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.80730819],
                      [0.        , -1.0859024 , -0.73246199,  0.        ,  2.08711336,  0.        , 0.        ],
                      [0.        ,  0.18103513,  1.17786194,  0.        ,  1.03111446, -1.08456791, -1.36347154]]
        e = rel_error(correct_dx, dx)
        print('dX relative difference:', e)
        self.assertTrue(e <= 5e-08)

    def runTest(self):
        self.test_relu_1_forward()
        self.test_relu_2_backward()

class TestTanh(unittest.TestCase):
    def test_tanh_1_forward(self):
        print('\n==================================')
        print('          Test tanh forward       ')
        print('==================================')
        x = np.linspace(-1.5, 1.5, num=30).reshape(6, 5)
        tanh = Tanh()
        out = tanh.forward(x)
        correct_out = [[-0.90514825, -0.88460402, -0.85993717, -0.83047189, -0.79548953],
                       [-0.75425761, -0.70607365, -0.65032522, -0.58656543, -0.51459914],
                       [-0.43457022, -0.34703513, -0.25300497, -0.15393885, -0.05167806],
                       [ 0.05167806,  0.15393885,  0.25300497,  0.34703513,  0.43457022],
                       [ 0.51459914,  0.58656543,  0.65032522,  0.70607365,  0.75425761],
                       [ 0.79548953,  0.83047189,  0.85993717,  0.88460402,  0.90514825]]
        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        # self.assertTrue(True)
        self.assertTrue(e <= 5e-8)

    def test_tanh_2_backward(self):
        print('\n==================================')
        print('          Test tanh backward      ')
        print('==================================')

        np.random.seed(123)
        x = np.random.randn(5, 5)
        tanh = Tanh()
        out = tanh.forward(x)
        dx = tanh.backward(x, 0.0)
        
        correct_dx = [[-0.39900528,  0.42055529,  0.26147548, -0.26911137, -0.42115391],
                      [ 0.22601191, -0.074565  , -0.35876431,  0.34549378, -0.44238431],
                      [-0.44192875, -0.0938645 ,  0.27373408, -0.43556065, -0.36680073],
                      [-0.36170986,  0.10450339,  0.10753923,  0.41907822,  0.33386819],
                      [ 0.44697633,  0.27393871, -0.43260586,  0.37333295, -0.349298  ]]
        e = rel_error(correct_dx, dx)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-8)
    
    def runTest(self):
        self.test_tanh_1_forward()
        self.test_tanh_2_backward()
