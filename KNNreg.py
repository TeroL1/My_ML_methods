from typing import Union
import numpy as np
import pandas as pd


class MyKNNReg():
    """
    Реализация KNN для регрессии.
    Поддерживает:
    - различное число соседей k
    - различные метрики (euclidean, chebyshev, manhattan, cosine)
    - различные виды учета веса соседей (uniform, rank, distance)
    """
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        
        self.train_size = None
        
    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = (X.shape[0], X.shape[1])
        
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.train_X = X
        self.train_y = y
       
    def predict(self, X: pd.DataFrame) -> np.array:
        data_len = X.shape[0]
        nearests = self._findNearestPointsY(X)
        
        preds = np.zeros(data_len)
        for index, nearest in enumerate(nearests):
            to_preds = self._computеWeight(nearest)
            
            preds[index] = to_preds
            
        return preds
    
        
    def _findNearestPointsY(self, X: pd.DataFrame) -> np.ndarray:
        data_len = X.shape[0]
        X = X.to_numpy()
        answer = np.zeros((data_len, self.k, 2))
        
        for index, vector in enumerate(X):
            nearest = []
            for train_index, train_vector in enumerate(self.train_X):
                pred = self.train_y[train_index]
                
                metric = self.findDistance(vector, train_vector)
                nearest.append((pred, metric))

            nearest.sort(key = lambda x: x[1])
            for pred in range(self.k):
                answer[index][pred][0], answer[index][pred][1] = nearest[pred][0], nearest[pred][1]
         
        return answer
    
    def _computеWeight(self, vec: np.array) -> (Union[int, float], Union[int, float]):
        return getattr(self, '_' + self.weight + 'Weight')(vec)
    
    @staticmethod 
    def _uniformWeight(vec: np.ndarray) -> (int, int):
        return np.mean((vec.T)[0])
    
    @staticmethod 
    def _rankWeight(vec: np.ndarray) -> (float, float):
        n = len(vec)
        ans = 0
        
        denominator = sum([1 / index for index in range(1, n + 1)])
        weight_preds = sum([(vec[index - 1][0] * (1 / index)) / denominator for index in range(1, n + 1)])
            
        return weight_preds
    
    @staticmethod 
    def _distanceWeight(vec: np.ndarray) -> (float, float):
        n = len(vec)
        ans = 0
        
        denominator = sum([1 / vec[index][1] for index in range(n)])
        weight_preds = sum([(vec[index][0] * (1 / vec[index][1])) / denominator for index in range(n)])
            
        return weight_preds
            
    @staticmethod        
    def _euclideanMetric(vec1: np.array, vec2: np.array) -> float:
        metric = np.sqrt(np.sum((vec1 - vec2) ** 2))
        
        return metric
    
    @staticmethod        
    def _chebyshevMetric(vec1: np.array, vec2: np.array) -> float:
        metric = max(np.abs(vec1 - vec2))
        
        return metric
    
    @staticmethod        
    def _manhattanMetric(vec1: np.array, vec2: np.array) -> float:
        metric = np.sum(np.abs(vec1 - vec2))
        
        return metric
    
    @staticmethod        
    def _cosineMetric(vec1: np.array, vec2: np.array) -> float:
        denominator = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
        numerator = np.sum(vec1 * vec2)
        
        metric = 1 - numerator / denominator
        
        return metric
    
    def findDistance(self, vec1: np.array, vec2: np.array) -> float:
        return getattr(self, '_' + self.metric + 'Metric')(vec1, vec2)
