import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def fit(self, x, y, epochs, batch_size, lr, optim):

        """
        The optimization of Logistic Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for logistic regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer. (fixed to 'stochastic gradient descent' for this assignment.)

        [Output]
            None

        """

        # ========================= EDIT HERE ========================
        num_data = x.shape[0]
        indexes = [i for i in range(0,num_data)]
        index = [indexes[i*batch_size:(i+1)*batch_size] for i in range((len(indexes) + batch_size - 1) // batch_size)]
        y = np.reshape(y,(-1,1))
        for i in range(epochs):
            for idx in index:
                grad = np.zeros((self.num_features,1))
                diff= (self._sigmoid(np.dot(x[idx],self.W))) - y[idx]
                grad = np.dot(np.transpose(x[idx]),diff)/len(idx)
                self.W = optim.update(self.W,grad,lr)
        # ============================================================
    
    def _sigmoid(self, x):
        """
        Apply sigmoid function to the given argument 'x'.

        [Inputs]
            x : Input of sigmoid function. Numpy array of arbitrary shape.

        [Output]
            sigmoid: Output of sigmoid function. Numpy array of same shape with 'x'.

        """
        sigmoid = None
        # ========================= EDIT HERE ========================
        ex = np.exp(x) 
        sigmoid = ex / (1+ex)

        # ============================================================
        return sigmoid

    def eval(self, x, threshold=0.5):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for logistic regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
                    Pred = 1 if probability > threshold 
                    Pred = 0 if probability <= threshold 
        """

        # ========================= EDIT HERE ========================
        pred = self._sigmoid(np.dot(x,self.W))
        pred = np.ravel(pred,order='C')
        #print(self.W)
        for i in range(pred.shape[0]):
            if(pred[i] > threshold):
                pred[i] = 1
            else:
                pred[i] = 0
        # ============================================================
        return pred
