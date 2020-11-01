import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.Y = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # calculating distances from given point to all the other points
    # sorting them and returns the most frequent label
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        labels = [self.Y[i] for i in k_indices]
        return Counter(labels).most_common(1)[0][0]
