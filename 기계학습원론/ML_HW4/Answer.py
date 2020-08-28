from collections import OrderedDict
import numpy as np

from utils import check_conv_validity, check_pool_validity

np.random.seed(123)

def softmax(z):
    # Numerically stable softmax.
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)
    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum
    return sm

def zero_pad(x, pad):
    ########################################################################
    # Zero padding
    # Given x and pad value, pad input 'x' around height & width.
    # 
    # [Input]
    # x: 4-D input batch data
    # - Shape : (# data, In Channel, Height, Width)
    #     
    # pad: pad value. how much to pad on one side.
    # e.g. pad=2 => pad 2 zeros on left, right, up & down.
    # 
    # [Output]
    # padded_x : padded x
    # - Shape : (Batch size, In Channel, Padded_Height, Padded_Width)
    ########################################################################

    padded_x = None
    N, C, H, W = x.shape
    # =============================== EDIT HERE ===============================
    padded_x = np.array([[np.pad(x[b, ic], pad_width=pad, mode='constant', constant_values=0) for ic in range(C)]for b in range(N)], dtype=np.float64)
    # =========================================================================
    return padded_x

class ReLU:
    #################################################
    # ReLU Function. ReLU(x) = max(0, x)
    # Implement forward & backward path of ReLU.
    # 
    # ReLU(x) = x if x > 0. 0 otherwise.
    # Be careful. It's '>', not '>='.
    # (ReLU in previous HW might be different.)
    #################################################

    def __init__(self):
        # 1 (True) if ReLU input <= 0
        self.zero_mask = None

    def forward(self, z):
        #################################################
        # ReLU Forward.
        # ReLU(x) = max(0, x)
        # 
        # z --> (ReLU) --> out
        # 
        # [Inputs]
        #     z : ReLU input in any shape.
        # 
        # [Outputs]
        #     out : ReLU(z).
        #################################################
        
        out = None
        
        # =============================== EDIT HERE ===============================
        out = z.copy()
        self.zero_mask = (z <= 0)
        np.maximum(z,0,out)
        # =========================================================================
        self.output_shape = out.shape
        return out

    def backward(self, d_prev, reg_lambda):
        #################################################
        # ReLU Backward.
        # 
        # z --> (ReLU) --> out
        # dz <-- (dReLU) <-- d_prev(dL/dout)
        # 
        # [Inputs]
        #     d_prev : Gradients flow from upper layer.
        #     reg_lambda: L2 regularization weight. (Not used in activation function)
        # 
        # [Outputs]
        #     dz : Gradients w.r.t. ReLU input z.
        #################################################
        dz = None
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        # =============================== EDIT HERE ===============================
        dz = d_prev
        dz[self.zero_mask] = 0
        # =========================================================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN ReLU
        pass

    def summary(self):
        return 'ReLU Activation'


class Sigmoid:
    #################################################
    # Sigmoid Function.
    # Implement forward & backward path of Sigmoid.
    #################################################

    def __init__(self):
        self.out = None

    def forward(self, z):
        #############################################################################
        # Sigmoid Forward.
        # 
        # z --> (Sigmoid) --> self.out
        # 
        # [Inputs]
        #     z : Sigmoid input in any shape.
        # 
        # [Outputs]
        #     self.out : Values applied elementwise sigmoid function on input 'z'.
        #############################################################################
        self.out = None
        # =============== EDIT HERE ===============
        ex = np.exp(z) 
        self.out = ex / (1+ex)
        # =========================================
        self.output_shape = self.out.shape
        return self.out

    def backward(self, d_prev, reg_lambda):
        #################################################################################
        # Sigmoid Backward.
        # 
        # z --> (Sigmoid) --> self.out
        # dz <-- (dSigmoid) <-- d_prev(dL/d self.out)
        # 
        # [Inputs]
        #     d_prev : Gradients flow from upper layer.
        #     reg_lambda: L2 regularization weight. (Not used in activation function)
        # 
        # [Outputs]
        #     dz : Gradients w.r.t. Sigmoid input z .
        #################################################################################

        dz = None
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        # =============== EDIT HERE ===============
        dz = np.multiply(np.multiply(d_prev, self.out), (1 - self.out))
        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Sigmoid
        pass

    def summary(self):
        return 'Sigmoid Activation'


class Tanh:
    #################################################
    # Hyperbolic Tangent Function(Tanh).
    # Implement forward & backward path of Tanh.
    #################################################

    def __init__(self):
        self.out = None

    def forward(self, z):
        #########################################################################
        # Hyperbolic Tangent Forward.
        # 
        # z --> (Tanh) --> self.out
        # 
        # [Inputs]
        #   z : Tanh input in any shape.
        # 
        # [Outputs]
        #     self.out : Values applied elementwise tanh function on input 'z'.
        # 
        # =====CAUTION!=====
        # You are not allowed to use np.tanh function!
        #########################################################################
        self.out = None
        # =============== EDIT HERE ===============
        ex = np.exp(z)
        self.out = (ex-(1/ex))/(ex+(1/ex))        
        # =========================================
        self.output_shape = self.out.shape
        return self.out

    def backward(self, d_prev, reg_lambda):
        ################################################################################
        # Hyperbolic Tangent Backward.
        # 
        # z --> (Tanh) --> self.out
        # dz <-- (dTanh) <-- d_prev(dL/d self.out)
        # 
        # [Inputs]
        #     d_prev : Gradients flow from upper layer.
        #     reg_lambda: L2 regularization weight. (Not used in activation function)
        # 
        # [Outputs]
        #     dz : Gradients w.r.t. Tanh input z .
        #     In other words, the derivative of tanh should be reflected on d_prev.
        ################################################################################
        dz = None
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        # =============== EDIT HERE ===============
        dz = np.multiply(d_prev,1-np.multiply(self.out,self.out))        
        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Tanh
        pass

    def summary(self):
        return 'Tanh Activation'


################################################################################################################
#    ** ConvolutionLayer **                                                                                    #
#   Single Convolution Layer.                                                                                  #
#                                                                                                              #
#   Given input images,                                                                                        #
#   'Convolution Layer' do convolution on input with Ws and convolution options (stride, pad ...).        #
#                                                                                                              #
#   You need to implement forward and backward pass of single convolution layer.                               #
#   (This is NOT an entire CNN model.)                                                                         #
#                                                                                                              #                                                                   #
################################################################################################################


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)

        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        ##################################################################################################
        # Convolution Layer Forward.
        # 
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # 
        # [Output]
        # conv : convolution result
        # - Shape : (Conv_Height, Conv_Width)
        # - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'W size', 'Stride'
        # 
        ##################################################################################################
        batch_size, in_channel, _, _ = x.shape
        conv = self.convolution(x, self.w, self.b, self.stride, self.pad)
        self.output_shape = conv.shape
        return conv

    def convolution(self, x, w, b, stride=1, pad=0):
        #########################################################################################################
        # Convolution Operation.
        # 
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # w: 4-D convolution filter
        # - Shape : (Out Channel, In Channel, Kernel Height, Kernel Width)
        # b: 1-D bias
        # - Shape : (Out Channel)
        # - default : None
        # stride : Stride size
        # - dtype : int
        # - default : 1
        # pad: pad value, how much to pad around
        # - dtype : int
        # - default : 0
        # 
        # [Output]
        # conv : convolution result
        # - Shape : (Batch size, Out Channel, Conv_Height, Conv_Width)
        # - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel Height', 'Kernel Width'
        #########################################################################################################
        
        # Check validity
        check_conv_validity(x, w, stride, pad)

        if pad > 0:
            x = zero_pad(x, pad)
        
        self.x = x

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        # =============================== EDIT HERE ===============================
        conv_W = int(np.ceil((W-WW+1) / stride))
        conv_H = int(np.ceil((H-HH+1) / stride))
        conv = np.zeros((N, F, conv_H, conv_W))
        for bsize in range(N):
            for ochannel in range(F):
                for ichannel in range(C):
                    temp = np.zeros((conv_H,conv_W))
                    for i in range(0, H-HH+1, stride):
                        for j in range(0, W-WW+1, stride):
                            temp[i//stride][j//stride] = (x[bsize][ichannel][i:i+HH,j:j+WW] * w[ochannel][ichannel]).sum()
                    conv[bsize][ochannel] += temp
            if type(b) is np.ndarray:
                for ochannel in range(F):
                    conv[bsize][ochannel] += b[ochannel]        
        # =========================================================================
        return conv

    def backward(self, d_prev, reg_lambda):
        ####################################################################
        # Convolution Layer Backward.
        # Compute derivatives w.r.t x, W, b (self.x, self.W, self.b)
        # 
        # [Input]
        #   d_prev: Gradients value so far in back-propagation process.
        #   reg_lambda: L2 regularization weight. (Not used in activation function)
        # 
        # [Output]
        #   self.dx : Gradient values of input x (self.x)
        #   - Shape : (Batch size, channel, Heigth, Width)
        ####################################################################
        N, C, H, W = self.x.shape
        F, _, HH, WW = self.w.shape
        _, _, H_filter, W_filter = self.output_shape

        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.dx = np.zeros_like(self.x)
        # =============================== EDIT HERE ===============================
        for bsize in range(N):
            for ochannel in range(F):
                for ichannel in range(C):
                    temp = np.zeros_like((self.dw[ochannel][ichannel]))
                    for i in range(0, H-d_prev.shape[2]+1, self.stride):
                        for j in range(0, W-d_prev.shape[3]+1, self.stride):
                            temp[i//self.stride][j//self.stride] = (self.x[bsize][ichannel][i:i+d_prev.shape[2],j:j+d_prev.shape[3]] * d_prev[bsize][ochannel]).sum()
                    self.dw[ochannel][ichannel] += temp + (reg_lambda*self.w[ochannel][ichannel])
                self.db[ochannel] += np.sum(d_prev[bsize,ochannel])

        for bsize in range(N):
            for ichannel in range(C):
                for ochannel in range(F):
                    for i in range(1,H-1):
                        for j in range(1,W-1):
                            for x in range(HH):
                                for y in range(WW):
                                    if ((i-x) >= 0 and (i-x) < H_filter) and ((j-y) >= 0 and (j-y) < W_filter): 
                                        self.dx[bsize,ichannel,i,j] += d_prev[bsize, ochannel,i-x, j-y] * self.w[ochannel, ichannel, x, y]
        
        self.dx = self.dx[:, :, self.pad:H-self.pad, self.pad:W-self.pad]
        # =========================================================================
        return self.dx

    def update(self, learning_rate):
        # Update weights
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Filter Size : ' + str(self.w.shape) + ' Stride : %d, Zero padding: %d' % (self.stride, self.pad)

################################################################################################################
#    ** Max-Pooling Layer **                                                                                   #
#   Single Max-Pooling Layer.                                                                                  #
#                                                                                                              #
#   Given input images,                                                                                        #
#   'Max-Pooling Layer' max_pool (or subsample) maximum value in certain region of input                       #
#                                                                                                              #
#   Implement forward and backward                                                                             # 
################################################################################################################

class MaxPoolingLayer:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        ##############################################################################################
        # Max-Pooling Layer Forward. Pool maximum value by striding W.
        # 
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # 
        # [Output]
        # max_pool : max_pool result
        # - Shape : (Batch size, Out Channel, Pool_Height, Pool_Width)
        # - Pool_Height & Pool_Width can be calculated using 'Height', 'Width', 'Kernel_size', 'Stride'
        ###############################################################################################
        max_pool = None
        N, C, H, W = x.shape
        check_pool_validity(x, self.kernel_size, self.stride)
        
        self.x = x
        # =============================== EDIT HERE ===============================
        #self.mask = np.zeros_like(x)
        self.mask = list()
        out_W = int(np.ceil((W-self.kernel_size+1) / self.stride))
        out_H = int(np.ceil((H-self.kernel_size+1) / self.stride))
        max_pool = np.zeros((N, C, out_H, out_W))
        for bsize in range(N):
            tttemp = list()
            for channel in range(C):
                temp = list()
                for i in range(0, H-self.kernel_size+1, self.stride):
                    for j in range(0, W-self.kernel_size+1, self.stride):
                        _max = np.max(x[bsize,channel,i:i+self.kernel_size,j:j+self.kernel_size])
                        max_pool[bsize,channel,i//self.stride,j//self.stride] = _max
                        idx = np.argmax(x[bsize,channel,i:i+self.kernel_size,j:j+self.kernel_size])
                        ttemp = list()
                        ttemp.append(i+(idx//self.kernel_size))
                        ttemp.append(j+(idx%self.kernel_size))
                        ttemp.append(idx)
                        temp.append(ttemp)
                tttemp.append(temp)
            self.mask.append(tttemp)
        # =========================================================================
        self.output_shape = max_pool.shape
        return max_pool

    def backward(self, d_prev, reg_lambda):
        ##############################################################################################
        # Max-Pooling Layer Backward.
        # In backward pass, max-pool distributes gradients to where it came from in forward pass.
        # 
        # [Input]
        #   d_prev: Gradients value so far in back-propagation process.
        #       - Shape can be varies since either Conv. layer or FC-layer can follow.
        #           (Batch_size, Channel, Height, Width) or (Batch_size, FC Dimension)
        #   reg_lambda: L2 regularization weight. (Not used in pooling layer)
        # 
        # [Output]
        #   dx : max_pool gradients
        #   - Shape : (batch_size, channel, height, width) - same shape as input x
        ##############################################################################################
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        N, C, H, W = d_prev.shape
        dx = np.zeros_like(self.x)

        # =============================== EDIT HERE ===============================        
        for bsize in range(N):
            for channel in range(C):
                #for p in self.mask[channel]:
                for i in range(len(self.mask[bsize][channel])):
                    dx[bsize][channel][self.mask[bsize][channel][i][0]][self.mask[bsize][channel][i][1]] += d_prev[bsize][channel][i//H][i%W]

        # =========================================================================
        return dx

    def update(self, learning_rate):
        # NOT USED IN MAX-POOL
        pass

    def summary(self):
        return 'Pooling Size : ' + str((self.kernel_size, self.kernel_size)) + ' Stride : %d' % (self.stride)

################################################################################################################
#    ** Fully-Connected Layer **                                                                               #
#   Single Fully-Connected Layer.                                                                              #
#                                                                                                              #
#   Given input features,                                                                                      #
#   FC layer linearly transform input with weights (self.W) & bias (self.b)                                                       #
#                                                                                                              #
#   You need to implement forward and backward pass                                                            #
#   This FC Layer works same as one in HW-4, so you can copy your codes if you need any.                    #
#                                                                                                              #
################################################################################################################

class FCLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.w = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim / 2)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        ######################################################
        # FC Layer Forward.
        # Use variables : self.x, self.W, self.b
        # 
        # [Input]
        # x: Input features.
        # - Shape : (Batch size, In Channel, Height, Width)
        # or
        # - Shape : (Batch size, input_dim)
        # 
        # [Output]
        # self.out : fc result
        # - Shape : (Batch size, output_dim)
        ######################################################
        # Flatten input if needed.
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        self.x = x
        # =============================== EDIT HERE ===============================
        self.out = np.dot(self.x,self.w)+self.b
        # =========================================================================
        return self.out


    def backward(self, d_prev, reg_lambda):
        ###########################################################################
        # FC Layer Backward.
        # Use variables : self.x, self.W
        # 
        # [Input]
        #   d_prev: Gradients value so far in back-propagation process.
        #   reg_lambda: L2 regularization weight.
        # 
        # [Output]
        #   dx : Gradients w.r.t input x
        #   - Shape : (batch_size, input_dim) - same shape as input x
        ###########################################################################
        
        dx = None           # Gradient w.r.t. input x
        self.dw = None      # Gradient w.r.t. weight (self.W)
        self.db = None      # Gradient w.r.t. bias (self.b)
        # =============================== EDIT HERE ===============================
        self.dw = np.dot(np.transpose(self.x),d_prev)+reg_lambda*self.w
        self.db = np.dot(np.ones(d_prev.shape[0]),d_prev)
        dx = np.dot(self.w,np.transpose(d_prev))
        dx = np.transpose(dx)
        # =========================================================================
        return dx


    def update(self, learning_rate):
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Input -> Hidden : %d -> %d ' % (self.w.shape[0], self.w.shape[1])

##################################################################################################
# ** Softmax Layer **
# Softmax Layer applies softmax (WITHOUT any weights or bias)
# 
# Given an score,
# 'SoftmaxLayer' applies softmax to make probability distribution. (Not log softmax or else...)
# 
# Implement forward, backward, and ce_loss
##################################################################################################

class SoftmaxLayer:
    def __init__(self):
        # No parameters
        pass

    def forward(self, x):
        ###########################################################################
        # Softmax Layer Forward.
        # Apply softmax (not log softmax or others...) on axis-1
        # 
        # Use 'softmax' function above in this file.
        # 
        # [Input]
        # x: Score to apply softmax
        # - Shape: (batch_size, # of class)
        # 
        # [Output]
        # y_hat: Softmax probability distribution.
        # - Shape: (batch_size, # of class)
        ###########################################################################
        self.y_hat = None
        # =============================== EDIT HERE ===============================
        self.y_hat = np.zeros_like(x)
        self.y_hat = softmax(x)
        # =========================================================================
        return self.y_hat

    def backward(self, d_prev=1, reg_lambda=0):
        ##############################################
        # Softmax Layer Backward.
        # Gradients w.r.t input score.
        # 
        # That is,
        # Forward  : softmax prob = softmax(score)
        # Backward : dL / dscore => 'dx'
        # 
        # Compute dx (dL / dscore).
        # Check loss function in the assignment document.
        # 
        # [Input]
        #   d_prev : Gradients flow from upper layer.
        #   reg_lambda: L2 regularization weight.
        # 
        # [Output]
        #   dx: Gradients of softmax layer input 'x'
        ##############################################
        batch_size = self.y.shape[0]
        dx = None
        # =============================== EDIT HERE ===============================
        dx = np.ones_like(self.y)
        dx = (self.y_hat-self.y)/batch_size
        # =========================================================================
        return dx

    def ce_loss(self, y_hat, y):
        ###################################################
        # Compute Cross-entropy Loss.
        # Use epsilon (eps) for numerical stability in log.
        # e.g. log(x + eps) to avoid zero input.

        # Check loss function in HW3 word file.

        # [Input]
        # y_hat: Probability after softmax.
        # - Shape : (batch_size, # of class)

        # y: One-hot true label
        # - Shape : (batch_size, # of class)

        # [Output]
        # self.loss : cross-entropy loss
        # - float
        ####################################################
        self.loss = None
        eps = 1e-10
        self.y_hat = y_hat
        self.y = y
        # =============================== EDIT HERE ===============================
        self.loss = 0
        N,C = y.shape[0],y.shape[1]
        for i in range(N):
            for j in range(C):
                if( y[i][j] == 1):
                    self.loss -= np.log(y_hat[i][j] + eps)
        self.loss /= N
        # =========================================================================
        return self.loss

    def update(self, learning_rate):
        # Not used in softmax layer.
        pass

    def summary(self):
        return 'Softmax layer'

###########################################################################
# ** CNN Classifier **
# 
# This is an class for entire CNN classifier.
# All the functions and variables are already implemented.
# Look at the codes below and see how the codes work.
# 
# <<< DO NOT CHANGE ANY THING BELOW>>>
###########################################################################
class CNN_Classifier:
    def __init__(self):
        self.layers = OrderedDict()
        self.softmax_layer = None
        self.loss = None
        self.pred = None

    def predict(self, x):
        # Outputs model softmax score
        for name, layer in self.layers.items():
            x = layer.forward(x)
        x = self.softmax_layer.forward(x)
        return x

    def forward(self, x, y, reg_lambda):
        # Predicts and Compute CE Loss
        reg_loss = 0
        self.pred = self.predict(x)
        ce_loss = self.softmax_layer.ce_loss(self.pred, y)

        for name, layer in self.layers.items():
            if isinstance(layer, FCLayer):
                norm = np.linalg.norm(layer.w, 2)
                reg_loss += 0.5 * reg_lambda *  norm * norm
            if isinstance(layer, ConvolutionLayer):
                norm = np.linalg.norm(layer.w.reshape(-1, 1), 2)
                reg_loss += 0.5 * reg_lambda *  norm * norm 

        self.loss = ce_loss + reg_loss

        return self.loss

    def backward(self, reg_lambda):
        # Back-propagation
        d_prev = 1
        d_prev = self.softmax_layer.backward(d_prev, reg_lambda)
        for name, layer in list(self.layers.items())[::-1]:
            d_prev = layer.backward(d_prev, reg_lambda)

    def update(self, learning_rate):
        # Update weights in every layer with dW, db
        for name, layer in self.layers.items():
            layer.update(learning_rate)

    def add_layer(self, name, layer):
        # Add Neural Net layer with name.
        if isinstance(layer, SoftmaxLayer):
            if self.softmax_layer is None:
                self.softmax_layer = layer
            else:
                raise ValueError('Softmax Layer already exists!')
        else:
            self.layers[name] = layer

    def summary(self):
        # Print model architecture.
        print('======= Model Summary =======')
        for name, layer in self.layers.items():
            print('[%s] ' % name + layer.summary())
        print('[Softmax Layer] ' + self.softmax_layer.summary())
        print()
