
import unittest
import numpy as np
from Answer import FCLayer
from utils import rel_error

class TestFCLayer(unittest.TestCase):
    def test_fc_layer_1_forward(self):
        print('\n==================================')
        print('       Test FC layer forward      ')
        print('==================================')
        np.random.seed(123)
        in_dim, out_dim = 5, 7
        x = np.random.randn(5, in_dim)
        fc_layer = FCLayer(in_dim, out_dim)
        
        fc_out = fc_layer.forward(x)
        correct_out = [[-2.61587108, -1.61266299, -0.0669421 , -2.54145585,  1.28682999,  1.1125263 , -0.68786543],
                       [ 4.32876618,  2.55437969, -2.24952545,  1.52993583, -3.05818339, -3.06667247, -0.76003621],
                       [-0.09513516, -1.28239277, -1.76747871, -1.4526619 ,  0.94121269,  0.3876754 ,  0.23826321],
                       [-0.57981942, -2.07002539,  0.17966919,  0.89441346,  0.64115475,  1.20433427,  0.8938306 ],
                       [ 0.27645953,  0.52337315,  1.11057732,  0.96039962, -3.09284442, -1.62932547, -1.67869927]]

        e = rel_error(correct_out, fc_out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-8)
        
    def test_fc_layer_2_backward(self):
        print('\n==================================')
        print('       Test FC layer backward     ')
        print('==================================')
        np.random.seed(123)
        in_dim, out_dim = 5, 7
        x = np.random.randn(5, in_dim)
        fc_layer = FCLayer(in_dim, out_dim)
        
        fc_out = fc_layer.forward(x)
        d_prev = np.random.randn(*fc_out.shape)
        dx = fc_layer.backward(d_prev, 0.01)
        dw = fc_layer.dw
        db = fc_layer.db
        
        correct_dx = [[-0.38227102,  1.17252013,  0.12788217, -4.28457506,  2.23774842],
                      [ 1.39190197,  1.971887  ,  0.19018213, -1.17233471, -0.07123409],
                      [ 0.3228319 ,  0.29698036, -1.09577096, -0.24408399, -0.85236351],
                      [ 3.38556938,  0.8984432 , -2.31575898, -0.52158556, -1.97062228],
                      [-1.37065069, -0.89799099, -0.33241037,  0.8274308 ,  0.10209643]]
        correct_dw = [[ 3.72943234, -4.14214785,  0.75059198,  1.50031574, -1.74298193,  3.31825892, -2.45157517],
                      [ 1.57658455,  7.50683015,  3.36883864, -5.0322751 ,  1.38427174, -1.71863439, -0.59066825],
                      [-6.74286032,  5.76269827,  0.07986016, -0.26893331, -3.03193173, -5.36704897,  0.01743584],
                      [ 4.00285798, -1.51490194,  2.20612597,  0.87921304, -3.53759088,  1.92435603, -3.85981001],
                      [-1.2963323 ,  0.55252352, -1.29937229,  1.89050352, -0.36306998, -2.51168386,  0.3259339 ]]
        correct_db = [-1.73065675,  2.46168233,  2.06794376, -2.56596612, -1.15623993,  0.80887778, -1.34400948]

        dx_e = rel_error(correct_dx, dx)
        dw_e = rel_error(correct_dw, dw)
        db_e = rel_error(correct_db, db)

        print('Relative difference dx:', dx_e)
        print('Relative difference dw:', dw_e)
        print('Relative difference db:', db_e)

        self.assertTrue(dx_e <= 5e-8)
        self.assertTrue(dw_e <= 5e-6)
        self.assertTrue(db_e <= 5e-8)

    def test_fc_layer_3_update(self):
        print('\n==================================')
        print('        Test fc layer update      ')
        print('==================================')
        np.random.seed(123)
        in_dim, out_dim = 5, 7
        x = np.random.randn(5, in_dim)
        fc_layer = FCLayer(in_dim, out_dim)

        before_w = np.array(fc_layer.w, copy=True)
        before_b = np.array(fc_layer.b, copy=True)
        
        fc_out = fc_layer.forward(x)
        d_prev = np.random.randn(*fc_out.shape)
        dx = fc_layer.backward(d_prev, 0.01)
        fc_layer.update(learning_rate=0.05)

        after_w = fc_layer.w
        after_b = fc_layer.b

        correct_before_w = [[-0.40334947,  0.5737037 , -0.90357701, -0.08858724, -0.54502165, -0.16166788, -1.76998316],
                            [-1.12041591, -0.44264123,  0.58657875, -0.10981685,  0.00179992,  0.43527026, -0.55626763],
                            [ 0.17938167, -0.50935851, -1.09267413, -0.24722674,  0.36290669,  0.21414252, -0.00748226],
                            [ 1.51306465,  0.26114858,  0.619007  ,  1.41552614, -0.81845142, -0.65698735, 1.10282044 ],
                            [-0.50473919,  0.01877332,  0.6762948 ,  0.56333218,  1.10988747,  0.94592841, 0.67634331]]
        correct_before_b = [0., 0., 0., 0., 0., 0., 0.]

        correct_after_w = [[-0.58982108,  0.78081109, -0.94110661, -0.16360302, -0.45787255, -0.32758083, -1.6474044 ],
                           [-1.19924514, -0.81798274,  0.41813681,  0.14179691, -0.06741367,  0.52120198, -0.52673421],
                           [ 0.51652469, -0.79749342, -1.09666714, -0.23378007,  0.51450328,  0.48249497, -0.00835405],
                           [ 1.31292175,  0.33689368,  0.5087007 ,  1.37156548, -0.64157188, -0.75320515,  1.29581094],
                           [-0.43992258, -0.00885285,  0.74126341,  0.46880701,  1.12804097,  1.0715126 ,  0.66004662]]
        correct_after_b = [ 0.08653284, -0.12308412, -0.10339719,  0.12829831,  0.057812 ,  -0.04044389, 0.06720047]

        before_w_e = rel_error(correct_before_w, before_w)
        before_b_e = rel_error(correct_before_b, before_b)
        after_w_e = rel_error(correct_after_w, after_w)
        after_b_e = rel_error(correct_after_b, after_b)

        print('Relative difference before_w:', before_w_e)
        print('Relative difference before_b:', before_b_e)
        print('Relative difference after_w :', after_w_e)        
        print('Relative difference after_b :', after_b_e)

        self.assertTrue(before_w_e <= 5e-6)
        self.assertTrue(before_b_e <= 1e-11)
        self.assertTrue(after_w_e <= 5e-7)
        self.assertTrue(after_b_e <= 5e-8)
    
    def runTest(self):
        self.test_fc_layer_1_forward()
        self.test_fc_layer_2_backward()
        self.test_fc_layer_3_update()
