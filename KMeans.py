import numpy as np

np.random.seed(42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)


class KMeansClustering:
    def __init__(self, k=5, max_iters=100):
        self.k = k
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.k)]

        self.centroids = []

    def predict(self, X):
        self.n_samples, self.n_features = X.shape
        self.X = X

        random_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_idx]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            centroids_old = self.centroids

            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    # creates clusters with given centroids
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # calculates distances and returns the point with the smallest distance
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, x) for x in centroids]
        return np.argmin(distances)

    # returns np array of new centroids, which are chosen via their mean
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster_idx], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    # returns true if non of the centroids moved
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0
    # returns labels of given cluster
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
