import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2020)


def main():
    x1_data = [4.57, 1.59, 4.56, -4.5, -2.33, -2.19, -0.83, -3.52, 2.54, -2.41, 3.2, -3.53, 0.95, -0.1, 2.06, 4.66,
               2.28, -3.57, 4.16, 1.91]
    x2_data = [-1.69, 3.11, -0.22, -1.81, -2.01, 1.8, -4.05, 2.33, 2.59, -4.72, -2.28, -0.05, -2.92, 4.83, -1.65, -1.97,
               -4.13, -2.51, 2.51, 0.85]

    X = np.zeros([len(x1_data), 2], dtype=np.float32)
    Y = np.ones(len(x1_data), dtype=np.int32)

    x1_threshold = np.mean(x1_data)

    X[:, 0], X[:, 1] = x1_data, x2_data
    Y[x1_data < x1_threshold] = -1

    pos_x = X[Y == 1]
    neg_x = X[Y == -1]

    w = np.random.rand(X.shape[1])

    t = - (w[0] / w[1])
    tan_x = (np.arange(20) - 10)
    tan_y = t * tan_x

    plt.scatter(pos_x[:, 0], pos_x[:, 1], c='r')
    plt.scatter(neg_x[:, 0], neg_x[:, 1], c='b')
    plt.plot(tan_x, tan_y, c='k')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
    
    # ========================= EDIT HERE =========================
    """
    Find the linear decision boundary on given data, 
    using the idea on Lecture material 'W02 Supervised Learning' page 90.
    """
    while(1):
        flag = 0
        for i in range(20):
            if(np.sign(X[i,0]*w[0]+X[i,1]*w[1])!= Y[i]):
                flag+=1
                if(X[i] in pos_x):
                    w[0] = w[0]+X[i,0]
                    w[1] = w[1]+X[i,1]
                else:
                    w[0] = w[0]-X[i,0]
                    w[1] = w[1]-X[i,1]
        if(flag == 0):
            break
    # =============================================================

    print(w)

    t = - (w[0] / w[1])
    tan_x = (np.arange(20) -10)
    tan_y = t * tan_x

    plt.scatter(pos_x[:, 0], pos_x[:, 1], c='r')
    plt.scatter(neg_x[:, 0], neg_x[:, 1], c='b')
    plt.plot(tan_x, tan_y, c='k')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


if __name__=='__main__':
    main()