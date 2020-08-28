
import unittest
import numpy as np
from Answer import SoftmaxLayer
from utils import rel_error

class TestSoftmaxLayer(unittest.TestCase):
    def setUp(self):
        self.softmax_layer = SoftmaxLayer()

    def test_softmax_layer_1_forward(self):
        print('\n==================================')
        print('    Test softmax layer forward    ')
        print('==================================')
        np.random.seed(123)
        x = np.random.randn(5, 5)
        
        softmax_out = self.softmax_layer.forward(x)
        correct_out = [[0.06546572, 0.52558012, 0.25727246, 0.04298548, 0.10869621],
                       [0.52561239, 0.00890353, 0.06564194, 0.3574744 , 0.04236773],
                       [0.07215113, 0.12940411, 0.63209441, 0.07509449, 0.09125586],
                       [0.02836542, 0.39760227, 0.39006297, 0.11953103, 0.06443832],
                       [0.20005518, 0.42494369, 0.03753939, 0.31014926, 0.02731249]]

        e = rel_error(correct_out, softmax_out)
        print('Relative difference:', e)
        self.assertTrue(e <= 1e-6)
        
        out_sum = np.sum(softmax_out)
        sum_e = out_sum - len(x)
        print('Softmax sum error:', sum_e)
        self.assertTrue(sum_e == 0)
        
    def test_softmax_layer_2_ce_loss(self):
        print('\n==================================')
        print('    Test softmax layer ce loss    ')
        print('==================================')
        np.random.seed(123)
        x = np.random.randn(5, 5)
        num_data, num_classes = x.shape
        
        y_hat = self.softmax_layer.forward(x)
        y = np.zeros_like(y_hat)
        y_labels = np.random.permutation(num_data)
        y[list(range(len(y_labels))), y_labels] = 1
        loss = self.softmax_layer.ce_loss(y_hat, y)
        
        correct_loss = 2.3052757961131616
        
        e = rel_error(correct_loss, loss)
        print('Relative difference:', e)
        self.assertTrue(e <= 1e-11)

    def test_softmax_layer_3_backward(self):
        print('\n==================================')
        print('    Test softmax layer backward   ')
        print('==================================')
        np.random.seed(123)
        x = np.random.randn(5, 5)
        num_data, num_classes = x.shape
        
        y_hat = self.softmax_layer.forward(x)
        y = np.zeros_like(y_hat)
        y_labels = np.random.permutation(num_data)
        y[list(range(len(y_labels))), y_labels] = 1
        loss = self.softmax_layer.ce_loss(y_hat, y)
        dx = self.softmax_layer.backward(d_prev=1)

        correct_dx = [[ 0.01309314,  0.10511602, -0.14854551,  0.0085971 ,  0.02173924],
                      [ 0.10512248,  0.00178071,  0.01312839,  0.07149488, -0.19152645],
                      [ 0.01443023,  0.02588082,  0.12641888, -0.1849811 ,  0.01825117],
                      [-0.19432692,  0.07952045,  0.07801259,  0.02390621,  0.01288766],
                      [ 0.04001104, -0.11501126,  0.00750788,  0.06202985,  0.0054625 ]]
                      
        e = rel_error(correct_dx, dx)
        print('Relative difference:', e)
        self.assertTrue(e <= 1e-6)
    
    def runTest(self):
        self.test_softmax_layer_1_forward()
        self.test_softmax_layer_2_ce_loss()
        self.test_softmax_layer_3_backward()
