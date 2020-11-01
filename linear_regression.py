import numpy as np


class LinearRegression:
    # constructor with alpha-learning rate, n_iters-number of iterations
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.theta = None

    # vectorized implementation of gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights)

            d_theta = (self.alpha / n_samples) * np.dot(X.T, (y_predicted - y))

            self.weights -= d_theta

    def predict(self, X):
        y_predicted = np.dot(X, self.weights)
        return y_predicted
