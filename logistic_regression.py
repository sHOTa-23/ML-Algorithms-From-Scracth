import numpy as np


def _sigmoid(X):
    return 1 / (1 + np.exp((-X)))


class LogisticRegression:
    # constructor with alpha-learning rate, n_iters-number of iterations and threshold
    def __init__(self, alpha=0.01, n_iters=1000, threshold=0.5):
        self.alpha = alpha
        self.n_iters = n_iters
        self.threshold = threshold
        self.theta = None

    # vectorized implementation of gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        for _ in range(self.n_iters):
            l_model = np.dot(X, self.theta)
            y_predicted = _sigmoid(l_model)

            d_theta = (self.alpha / n_samples) * np.dot(X.T, (y_predicted - y))

            self.theta -= d_theta

    # calculating probabilities and then thresholding them
    def predict(self, X):
        linear_model = np.dot(X, self.theta)
        y_predicted = _sigmoid(linear_model)
        y_predicted_cls = [1 if i > self.threshold else 0 for i in y_predicted]
        return y_predicted_cls
