import numpy as np

class myLinearRegression:

    def __init__(self):
        self.w = np.array([])

    def fit(self, x, y):
        ones = np.ones([len(x), 1])
        x_1 = np.hstack((ones, x))
        self.w = np.linalg.inv(x_1.T.dot(x_1)).dot(x_1.T).dot(y)
    
    def predict(self, x):
        ones = np.ones([len(x), 1])
        x_1 = np.hstack((ones, x))
        return x_1.dot(self.w)