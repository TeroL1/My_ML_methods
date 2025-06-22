import numpy as np
import pandas as pd

class MyAgglomerative():
    def __init__(self, n_clusters = 3, metric = 'euclidean') -> None:
        self.n_clusters = n_clusters
        self.metric = metric

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit_predict(self, X: pd.DataFrame) -> np.array:
        clusters, distances = self._buildDistMatrix(X)
        n = len(distances)

        while n > self.n_clusters:
            clusters, distances = self._rebuildDistMatrix(X, clusters, distances)
            n = len(distances)

        prediction_len = X.shape[0]
        prediction = __class__._indexesInPrediction(clusters, prediction_len)
        return prediction

    def _buildDistMatrix(self, X: pd.DataFrame) -> tuple:
        n = X.shape[0]

        clusters = dict()
        distances = np.zeros((n, n))

        for index, vector in X.iterrows():
            distances[index, :] = self._findDistanceVector(X.to_numpy(), vector.to_numpy())
            distances[index, index] = float('inf')

            clusters[index] = set()
            clusters[index].add(index)

        return clusters, distances

    def _rebuildDistMatrix(self, X: pd.DataFrame, clusters: dict, distances: np.ndarray) -> tuple:
        min_index = np.argmin(distances)
        i, j = np.unravel_index(min_index, distances.shape)

        keys = list(clusters.keys())
        key_i, key_j = keys[i], keys[j]
        new_cluster = clusters[key_i] | clusters[key_j]
        del clusters[key_i]
        del clusters[key_j]

        new_key = max(clusters.keys(), default = -1) + 1
        clusters[new_key] = new_cluster

        new_keys = list(clusters.keys())
        n = len(new_keys)
        new_distances = np.full((n, n), float('inf'))

        for i in range(n):
            for j in range(i + 1, n):
                cluster_i = list(clusters[new_keys[i]])
                cluster_j = list(clusters[new_keys[j]])
                centroid_i = X.iloc[cluster_i].mean().to_numpy()
                centroid_j = X.iloc[cluster_j].mean().to_numpy()
                dist = self._findDistance(centroid_i, centroid_j)
                new_distances[i, j] = dist
                new_distances[j, i] = dist

        return clusters, new_distances

    @staticmethod
    def _indexesInPrediction(clusters: dict, prediction_len: int) -> np.array:
        prediction = np.zeros(prediction_len)

        for key, values in clusters.items():
            for value in values:
                prediction[value] = key

        return prediction

    @staticmethod
    def _euclideanMetric(vec1: np.array, vec2: np.array) -> float:
        metric = np.sqrt(np.sum((vec1 - vec2) ** 2))

        return metric

    @staticmethod
    def _euclideanMetricVector(matrix: np.ndarray, vec: np.array) -> np.array:
        metric = np.sqrt(np.sum((matrix - vec) ** 2, axis = 1))

        return metric

    @staticmethod
    def _chebyshevMetric(vec1: np.array, vec2: np.array) -> float:
        metric = np.max(np.abs(vec1 - vec2))

        return metric

    @staticmethod
    def _chebyshevMetricVector(matrix: np.ndarray, vec: np.array) -> np.array:
        metric = np.max(np.abs(matrix - vec), axis = 1)

        return metric

    @staticmethod
    def _manhattanMetric(vec1: np.array, vec2: np.array) -> float:
        metric = np.sum(np.abs(vec1 - vec2))

        return metric

    @staticmethod
    def _manhattanMetricVector(matrix: np.ndarray, vec: np.array) -> np.array:
        metric = np.sum(np.abs(matrix - vec), axis = 1)

        return metric

    @staticmethod
    def _cosineMetric(vec1: np.array, vec2: np.array) -> float:
        denominator = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
        numerator = np.sum(vec1 * vec2)

        metric = 1 - numerator / denominator

        return metric

    @staticmethod
    def _cosineMetricVector(matrix: np.ndarray, vec: np.array) -> np.array:
        denominator = np.sqrt(np.sum(matrix ** 2, axis = 1) * np.sum(vec ** 2))
        numerator = np.sum(matrix * vec, axis = 1)

        metric = 1 - numerator / denominator

        return metric

    def _findDistance(self, vec1: np.array, vec2: np.array) -> float:
        return getattr(self, '_' + self.metric + 'Metric')(vec1, vec2)

    def _findDistanceVector(self, matrix: np.ndarray, vec: np.array) -> np.array:
        return getattr(self, '_' + self.metric + 'Metric' + 'Vector')(matrix, vec)
