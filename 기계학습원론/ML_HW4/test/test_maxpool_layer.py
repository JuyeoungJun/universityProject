
import unittest
import numpy as np
from Answer import MaxPoolingLayer
from utils import rel_error

class TestMaxPoolingLayer(unittest.TestCase):
    def test_maxpool_layer_1_forward(self):
        print('\n==================================')
        print('       Test maxpool layer forward      ')
        print('==================================')
        np.random.seed(123)
        in_dim, out_dim = 3, 3
        kernel_size, stride = 3, 1

        x = np.random.randn(1, in_dim, 5, 5)
        maxpool_layer = MaxPoolingLayer(kernel_size, stride)
        
        maxpool_out = maxpool_layer.forward(x)

        correct_out = [[[[1.65143654, 1.49138963, 1.49138963],
                        [2.20593008, 2.20593008, 2.18678609],
                        [2.20593008, 2.20593008, 2.18678609]],

                        [[0.9071052 , 0.9071052 , 0.92746243],
                        [0.68822271, 0.68822271, 0.92746243],
                        [2.39236527, 2.39236527, 2.23814334]],

                        [[1.75488618, 1.75488618, 1.75488618],
                        [1.75488618, 1.75488618, 1.75488618],
                        [1.16220405, 1.16220405, 1.41729905]]]]

        e = rel_error(correct_out, maxpool_out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-8)
        
    def test_maxpool_layer_2_backward(self):
        print('\n==================================')
        print('       Test maxpool layer backward     ')
        print('==================================')
        np.random.seed(123)
        in_dim, out_dim = 3, 3
        kernel_size, stride = 3, 1

        x = np.random.randn(1, in_dim, 5, 5)
        maxpool_layer = MaxPoolingLayer(kernel_size, stride)
        
        maxpool_out = maxpool_layer.forward(x)
        d_prev = np.random.randn(*maxpool_out.shape)
        dx = maxpool_layer.backward(d_prev, 0.01)
        
        correct_dx = [[[[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 1.03972709,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        , -0.52939562,  0.        ,  0.        ],
                        [ 0.        , -1.47139598,  2.06254556,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]],

                        [[ 0.        , -1.40066055,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        , -0.56802076],
                        [ 0.        ,  0.        ,  0.87459023,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 0.        , -0.11714651,  0.        ,  0.        ,  0.18103513]],

                        [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        , -0.19467321,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        , -1.97788793],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.26287825,  0.        ,  0.        ]]]]

        dx_e = rel_error(correct_dx, dx)

        print('Relative difference dx:', dx_e)

        self.assertTrue(dx_e <= 5e-8)

    
    def runTest(self):
        self.test_maxpool_layer_1_forward()
        self.test_maxpool_layer_2_backward()
