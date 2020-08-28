import unittest
import numpy as np
from Answer import FCLayer, SoftmaxLayer, Sigmoid, Tanh, ReLU, ClassifierModel
from utils import rel_error

def test_model(num_feat, num_classes):
    classifier = ClassifierModel()
    classifier.add_layer('FC-1', FCLayer(num_feat, 2))
    classifier.add_layer('Sigmoid', Sigmoid())
    classifier.add_layer('FC-2', FCLayer(2, 5))
    classifier.add_layer('ReLU', ReLU())
    classifier.add_layer('FC-3', FCLayer(5, 3))
    classifier.add_layer('tanh', Tanh())
    classifier.add_layer('FC-4', FCLayer(3, num_classes))
    classifier.add_layer('Softmax', SoftmaxLayer())
    return classifier

class TestClassifier(unittest.TestCase):
    def test_classifier_1_predict(self):
        print('\n==================================')
        print('      Test classifier predict     ')
        print('==================================')
        np.random.seed(123)
        num_feat, num_classes = 3, 7

        classifier = test_model(num_feat, num_classes)

        x = np.random.randn(3, num_feat)
        out = classifier.predict(x)
        
        correct_out = [[0.03012191, 0.34280256, 0.07083601, 0.1046846 , 0.34188104, 0.04817216, 0.06150172],
                       [0.0314701 , 0.34068337, 0.07228203, 0.10594079, 0.33863153, 0.04890545, 0.06208673],
                       [0.14932804, 0.18083691, 0.12796463, 0.09012277, 0.13528885, 0.16704287, 0.14941594]]

        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-8)
        
    def test_classifier_2_forward(self):
        print('\n==================================')
        print('      Test classifier forward     ')
        print('==================================')
        np.random.seed(123)
        num_feat, num_classes = 3, 7

        classifier = test_model(num_feat, num_classes)

        x = np.random.randn(3, num_feat)
        y = np.zeros((3, num_classes))
        labels = np.random.permutation(3)
        y[list(range(len(labels))), labels] = 1.0

        ce_loss = classifier.forward(x, y, 0.001)

        correct_ce_loss = 1.8824037412695567

        e = rel_error(correct_ce_loss, ce_loss)
        
        print('Relative difference:', e)

        self.assertTrue(e <= 5e-8)

    def test_classifier_3_backward(self):
        print('\n==================================')
        print('      Test classifier backward    ')
        print('==================================')
        np.random.seed(123)
        num_feat, num_classes = 3, 7

        classifier = test_model(num_feat, num_classes)

        x = np.random.randn(3, num_feat)
        y = np.zeros((3, num_classes))
        labels = np.random.permutation(3)
        y[list(range(len(labels))), labels] = 1.0

        ce_loss = classifier.forward(x, y, 0.01)
        classifier.backward(0.01)
        
        fc_3_dW = np.array(classifier.layers['FC-3'].dW, copy=True)

        classifier.update(learning_rate=0.01)

        fc_1_W = np.array(classifier.layers['FC-1'].W, copy=True)
        
        correct_fc_3_dW = [[ 0.01395153,  0.01383045,  0.00635019],
                           [ 0.49941591,  0.04122437,  0.08229664],
                           [ 0.14433844,  0.04878744,  0.02870912],
                           [-0.00403349,  0.00573704, -0.00903577],
                           [-0.00088587, -0.00545022, -0.00161668]]
        correct_fc_1_W = [[-0.88438284,  0.81210145],
                          [ 0.23247273, -1.23141275],
                          [-0.47319641,  1.34893614]]

        fc_3_dW_e = rel_error(correct_fc_3_dW, fc_3_dW)
        fc_1_W_e = rel_error(correct_fc_1_W, fc_1_W)
        
        print('Relative difference fc_3_dW_e:', fc_3_dW_e)
        print('Relative difference fc_1_W_e:', fc_1_W_e)

        self.assertTrue(fc_3_dW_e <= 5e-6)
        self.assertTrue(fc_1_W_e <= 5e-8)
    
    def runTest(self):
        self.test_classifier_1_predict()
        self.test_classifier_2_forward()
        self.test_classifier_3_backward()
