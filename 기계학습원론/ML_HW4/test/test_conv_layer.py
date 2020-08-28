import unittest
import numpy as np
from Answer import ConvolutionLayer, MaxPoolingLayer, FCLayer, SoftmaxLayer, Sigmoid, Tanh, ReLU, CNN_Classifier
from utils import rel_error

def test_model(num_feat, num_classes):
    classifier = CNN_Classifier()
    classifier.add_layer('Conv-1', ConvolutionLayer(num_feat, 2, kernel_size=3, stride=1, pad=1))
    classifier.add_layer('ReLU', ReLU())
    classifier.add_layer('Conv-2', ConvolutionLayer(2, 3, kernel_size=3, stride=1, pad=1))
    classifier.add_layer('tanh', Tanh())
    classifier.add_layer('Conv-3', ConvolutionLayer(3, 3, kernel_size=3, stride=1, pad=0))
    classifier.add_layer('Sigmoid', Sigmoid())
    classifier.add_layer('Max-pool - 1', MaxPoolingLayer(kernel_size=2, stride=1))
    classifier.add_layer('FC-4', FCLayer(12, num_classes))
    classifier.add_layer('Softmax', SoftmaxLayer())
    return classifier

class TestClassifier(unittest.TestCase):
    def test_classifier_1_predict(self):
        print('\n==================================')
        print('      Test classifier predict     ')
        print('==================================')
        np.random.seed(123)
        in_dim, num_classes = 1, 7

        classifier = test_model(in_dim, num_classes)
        x = np.random.randn(5, in_dim, 5, 5)
        out = classifier.predict(x)

        correct_out = [[0.02164869, 0.66709033, 0.03838546, 0.0290932 , 0.00573851, 0.01742173, 0.22062208],
                        [0.04049546, 0.61548575, 0.06589824, 0.04041239, 0.00798417, 0.03396838, 0.19575561],
                        [0.0241649 , 0.68671605, 0.03490482, 0.0241778 , 0.00504781, 0.01341441, 0.21157421],
                        [0.02155642, 0.66083457, 0.03907124, 0.02851615, 0.00536027, 0.01696291, 0.22769844],
                        [0.02336113, 0.74348208, 0.04048574, 0.02411253, 0.00481644, 0.02555465, 0.13818742]]

        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-6)
        
    def test_classifier_2_forward(self):
        print('\n==================================')
        print('      Test classifier forward     ')
        print('==================================')
        np.random.seed(123)
        num_feat, num_classes = 3, 7

        classifier = test_model(num_feat, num_classes)

        x = np.random.randn(3, num_feat, 5, 5)
        y = np.zeros((3, num_classes))
        labels = np.random.permutation(3)
        y[list(range(len(labels))), labels] = 1.0

        ce_loss = classifier.forward(x, y, 0.001)
        correct_ce_loss = 2.7128840564513546

        e = rel_error(correct_ce_loss, ce_loss)
        
        print('Relative difference:', e)

        self.assertTrue(e <= 5e-9)

    def test_classifier_3_backward(self):
        print('\n==================================')
        print('      Test classifier backward    ')
        print('==================================')
        np.random.seed(123)
        num_feat, num_classes = 3, 7

        classifier = test_model(num_feat, num_classes)

        x = np.random.randn(3, num_feat, 5, 5)
        y = np.zeros((3, num_classes))
        labels = np.random.permutation(3)
        y[list(range(len(labels))), labels] = 1.0

        ce_loss = classifier.forward(x, y, 0.01)
        classifier.backward(0.01)
        
        conv_3_W = np.array(classifier.layers['Conv-3'].w, copy=True)

        classifier.update(learning_rate=0.01)

        correct_conv_3_W = [[[[ 0.42614664, -1.60540974, -0.4276796 ],
                            [ 1.24286955, -0.73521696,  0.50124899],
                            [ 1.01273905,  0.27874086, -1.37094847]],

                            [[-0.33247528,  1.95941134, -2.02504576],
                            [-0.27578601, -0.55210807,  0.12074736],
                            [ 0.74821562,  1.60869097, -0.27023239]],

                            [[ 0.81234133,  0.49974014,  0.4743473 ],
                            [-0.56392393, -0.99732147, -1.10004311],
                            [-0.75643721,  0.32168658,  0.76094939]]],


                            [[[ 0.32346885, -0.5489551 ,  1.80597011],
                            [ 1.51886562, -0.35400011, -0.82343141],
                            [ 0.13021495,  1.26729865,  0.33276498]],

                            [[ 0.5565487 , -0.21208012,  0.4562709 ],
                            [ 1.54454445, -0.23966878,  0.14330773],
                            [ 0.25381648,  0.28372536, -1.41188888]],

                            [[-1.87686866, -1.01965507,  0.1679423 ],
                            [ 0.55385617, -0.53067456,  1.37725748],
                            [-0.14317597,  0.020316  , -0.19396387]]],


                            [[[ 0.13402679,  0.70447407,  0.66565344],
                            [-0.89842294,  1.52366378, -1.09502646],
                            [ 0.07922701, -0.27439657, -1.04899168]],

                            [[-0.07512059, -0.74081377,  0.07290724],
                            [ 0.40308596,  1.47192937,  0.30738422],
                            [-0.61122534, -0.39161981,  0.13997811]],

                            [[ 0.09346083,  1.45958927,  1.39535293],
                            [-0.35893593, -0.54864213, -2.5570546 ],
                            [-0.54892041, -0.97805771, -0.35482446]]]]

        conv_3_W_e = rel_error(correct_conv_3_W, conv_3_W)
        
        print('Relative difference conv_3_W_e:', conv_3_W_e)

        self.assertTrue(conv_3_W_e <= 5e-7)
    
    def runTest(self):
        self.test_classifier_1_predict()
        self.test_classifier_2_forward()
        self.test_classifier_3_backward()
