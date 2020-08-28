import unittest

from test.test_activation import TestSigmoid, TestReLU, TestTanh
from test.test_softmax_layer import TestSoftmaxLayer
from test.test_fc_layer import TestFCLayer
from test.test_classifier import TestClassifier
from test.test_conv_layer import TestConvolutionLayer
from test.test_maxpool_layer import TestMaxPoolingLayer

"""

Here, we provide codes to check your implementation.
Each 'Test*' class check whether your implementation yields correct answers within acceptable error range.
You can check part of your implementation by only giving test instances you want in variable 'test_cases'.
By default, the codes will check all 6 part of your answer.

"""
test_cases = [TestSigmoid(), TestReLU(), TestTanh(), TestSoftmaxLayer(), TestFCLayer(), TestConvolutionLayer(), TestMaxPoolingLayer(), TestClassifier()]

if __name__ == '__main__':
    suite = unittest.TestSuite(test_cases)
    unittest.TextTestRunner().run(suite)
