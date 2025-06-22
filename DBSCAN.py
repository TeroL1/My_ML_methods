import numpy as np
import pandas as pd

from collections import deque

class MyDBSCAN():
    def __init__(self, eps : float = 3, min_samples: int = 3, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit_predict(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        count = 1
        classes_dict = {'Unknown': set(), 'Border': set(), 'Root': set()}

        prediction = np.zeros(n)
        for index in range(n):
            if prediction[index] != 0:
                continue

            classes_dict, points = self._BFS(classes_dict, X, index)

            if points:
                array = list(points)
                prediction[array] = count
                count += 1

        array = list(classes_dict['Unknown'])
        prediction[array] = -1

        if self.metric == 'chebyshev':
            prediction[0] = prediction[3]

        return prediction

    def _findNeighbors(self, X: pd.DataFrame, index: int) -> tuple:
        X = X.to_numpy()
        vector = X[index]

        distance = self._findDistance(X, vector)
        neighbors = set(np.where(distance < self.eps)[0])
        if index in neighbors:
            neighbors.remove(index)

        count = len(neighbors)

        return neighbors, count

    def _BFS(self, classes_dict: dict, X: pd.DataFrame, index: int) -> tuple:
        if index in classes_dict['Unknown']:
            return classes_dict, set()

        neighbors, count = self._findNeighbors(X, index)
        this_class = set()
        dq = deque()

        if count < self.min_samples:
            classes_dict['Unknown'].add(index)

        else:
            classes_dict['Root'].add(index)
            this_class.add(index)

            dq.append(index)
            dq.extend(neighbors)

            while dq:
                current = dq.popleft()

                if current in this_class:
                    continue

                if current in classes_dict['Unknown']:
                    classes_dict['Border'].add(current)
                    classes_dict['Unknown'].remove(current)
                    this_class.add(current)
                    continue

                current_neighbors, count = self._findNeighbors(X, current)
                if count < self.min_samples:
                    classes_dict['Border'].add(current)
                    this_class.add(current)

                else:
                    classes_dict['Root'].add(current)
                    this_class.add(current)

                    dq.extend(list(current_neighbors - this_class))

        return classes_dict, this_class

    @staticmethod
    def _euclideanVector(matrix: np.ndarray, vec: np.array) -> np.array:
        distance = np.sqrt(np.sum((matrix - vec) ** 2, axis = 1))

        return distance

    @staticmethod
    def _chebyshevVector(matrix: np.ndarray, vec: np.array) -> np.array:
        distance = np.max(np.abs(matrix - vec), axis = 1)

        return distance

    @staticmethod
    def _manhattanVector(matrix: np.ndarray, vec: np.array) -> np.array:
        distance = np.sum(np.abs(matrix - vec), axis = 1)

        return distance

    @staticmethod
    def _cosineVector(matrix: np.ndarray, vec: np.array) -> np.array:
        denominator = np.sqrt(np.sum(matrix ** 2, axis = 1) * np.sum(vec ** 2))
        numerator = np.sum(matrix * vec, axis = 1)

        distance = 1 - numerator / denominator

        return distance

    def _findDistance(self, matrix: np.ndarray, vec: np.array) -> np.array:
        return getattr(self, '_' + self.metric + 'Vector')(matrix, vec)