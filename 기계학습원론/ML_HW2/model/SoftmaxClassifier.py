import time
import numpy as np

class SoftmaxClassifier:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.zeros((self.num_features, self.num_label))

    def train(self, x, y, epochs, batch_size, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        y : (N, )
        epochs: (int) # of training epoch to execute
        batch_size : (int) # of minibatch size
        lr : (float), learning rate
        optimizer : (Class) optimizer to use

        Returns:
        None

        Description:
        Given training data, hyperparameters and optimizer, execute training procedure.
        Weight should be updated by minibatch (not the whole data at a time)
        Procedure for one epoch is as follow:
        - For each minibatch
            - Compute probability of each class for data
            - Compute softmax loss
            - Compute gradient of weight
            - Update weight using optimizer
        * loss of one epoch = refer to the loss function in the instruction.
        """
        num_data, num_feat = x.shape
        num_batches = int(np.ceil(num_data / batch_size))

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            # ========================= EDIT HERE ========================
            indexes = [i for i in range(0,num_data)]
            index = [indexes[i*batch_size:(i+1)*batch_size] for i in range((len(indexes) + batch_size - 1) // batch_size)]
            for idx in index:
                prob, epoch_loss = self.forward(x[idx],y[idx])
                grad_weight = self.compute_grad(x[idx],y[idx],self.W,prob)
                self.W = optimizer.update(self.W,grad_weight,lr)
               
            # ============================================================
            epoch_elapsed = time.time() - epoch_start
            print('epoch %d, loss %.4f, time %.4f sec.' % (epoch, epoch_loss, epoch_elapsed))

    def forward(self, x, y):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data 
        y : (N, ), label for each data

        Returns:
        prob: (N, C), probability distribution over classes for N data
        softmax_loss : float, softmax loss for N input

        Description:
        Given N data and their labels, compute softmax probability distribution and loss.
        """
        num_data, num_feat = x.shape
        _, num_label = self.W.shape
        
        prob = None
        softmax_loss = 0.0
        # ========================= EDIT HERE ========================
        prob = self._softmax(np.dot(x,self.W))
        softmax_loss = -np.sum(np.log(prob))
        softmax_loss /= num_data
        # ============================================================
        return prob, softmax_loss

    def compute_grad(self, x, y, weight, prob):
        """
        N : # of minibatch data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        weight : (D, C), Weight matrix of classifier
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data. (0 <= c < C for c in label)

        Returns:
        gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

        Description:
        Given input (x), weight, probability and label, compute gradient of weight.
        """
        num_data, num_feat = x.shape
        _, num_class = weight.shape

        grad_weight = np.zeros_like(weight, dtype=np.float32)
        # ========================= EDIT HERE ========================
        for c in range(num_class):
            for i in range(num_data):
                diff = prob[i,c] - (1 if y[i] == c else 0)
                grad_weight[:, c] += (x[i, :].transpose())*diff
            grad_weight[:, c] /= num_data        
        # ============================================================
        return grad_weight


    def _softmax(self, x):
        """
        Inputs:
        x : (N, C), score before softmax

        Returns:
        softmax : (same shape with x), softmax distribution over axis-1

        Description:
        Given an input x, apply softmax funciton over axis-1.
        """
        softmax = None
        # ========================= EDIT HERE ========================
        N,C = x.shape
        softmax = np.copy(x)

        for i in range(N):
            exp_x = np.exp(x[i]-np.max(x[i]))
            softmax[i] = exp_x/np.sum(exp_x)
        # ============================================================
        return softmax
    
    def eval(self, x):
        """

        Inputs:
        x : (N, D), input data

        Returns:
        pred : (N, ), predicted label for N test data

        Description:
        Given N test data, compute probability and make predictions for each data.
        """
        pred, prob = None, None
        # ========================= EDIT HERE ========================
        N, D = x.shape
        prob = self._softmax(np.dot(x,self.W))  
        pred = np.zeros((N,))
        for i in range(N):
            pred[i] = list(prob[i]).index(prob[i].max())
        # ============================================================
        return pred, prob
