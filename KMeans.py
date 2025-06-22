import numpy as np
import pandas as pd

class MyKMeans():
    def __init__(self, n_clusters : int = 3, max_iter: int = 10, n_init: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = float('inf')

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame) -> None:
        np.random.seed(self.random_state)

        for epoch in range(self.n_init):
            centroids = self._precomputeClusters(X)

            for _ in range(self.max_iter):
                new_centroids = self._computeClusters(centroids, X)

                if np.allclose(centroids, new_centroids):
                    break

                centroids = new_centroids

            WCSS = self._computeWCSS(X, centroids)

            if WCSS < self.inertia_:
                self.inertia_ = WCSS
                self.cluster_centers_ = centroids

    def predict(self, X: pd.DataFrame) -> list:
        prediction = []

        for _, row in X.iterrows():
            nearest = __class__._nearestCentroid(row.values, self.cluster_centers_)
            prediction.append(nearest)

        return prediction

    def _precomputeClusters(self, X: pd.DataFrame) -> np.ndarray:
        min_vals = X.min().values
        max_vals = X.max().values
        centroids = np.random.uniform(low=min_vals, high=max_vals, size=(self.n_clusters, X.shape[1]))
        return centroids

    def _computeClusters(self, centroids: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        features_cnt = X.shape[1]
        indexes = self._findClusters(centroids, X)

        new_centroids = np.zeros((self.n_clusters, features_cnt))

        for cluster_id in range(self.n_clusters):
            if indexes[cluster_id]:
                X_cluster = X.iloc[indexes[cluster_id]]
                new_centroids[cluster_id, :] = X_cluster.mean(axis=0).values

            else:
                new_centroids[cluster_id, :] = centroids[cluster_id, :]

        return new_centroids

    def _findClusters(self, centroids: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        indexes = [[] for _ in range(self.n_clusters)]

        for index, row in X.iterrows():
            nearest = __class__._nearestCentroid(row.values, centroids)
            indexes[nearest].append(index)

        return indexes

    @staticmethod
    def _euclideanMetricVector(matrix: np.ndarray, vec: np.array) -> np.array:
        metric = np.sqrt(np.sum((matrix - vec) ** 2, axis = 1))

        return metric

    @staticmethod
    def _nearestCentroid(vec: np.array, centroids: np.ndarray) -> int:
        index = np.argmin(__class__._euclideanMetricVector(centroids, vec))

        return index

    def _computeWCSS(self, X: pd.DataFrame, centroids: np.ndarray) -> float:
        WCSS = 0

        indexes = self._findClusters(centroids, X)

        for cluster_id in range(self.n_clusters):
            if indexes[cluster_id]:
                X_cluster = X.iloc[indexes[cluster_id]].values
                cluster_metric = __class__._euclideanMetricVector(X_cluster, centroids[cluster_id])
                WCSS += np.sum(cluster_metric ** 2)

        return WCSS