import numpy as np
import pandas as pd
import random
from typing import Union

class MySVM():
    """
    Реализация SVM.
    Поддерживает:
    - выбор параметра C
    - стохастический градиентный спуск
    """
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001, C: float = 1, sgd_sample: Union[int, float] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.C = C
        
        self.sgd_sample = sgd_sample
        
        self.random_state = random_state
        
        self._weights = None
        self._b = None
        
    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        random.seed(self.random_state)
        
        _, features_len = X.shape[0], X.shape[1]
        X = X.to_numpy()
        y = y.to_numpy()
        
        y = (y - 0.5) * 2
        
        self._weights = np.ones(features_len)
        self._b = 1
        
        for step in range(self.n_iter):
            X_batch, y_batch = self._findBatches(X, y)
            data_len = X_batch.shape[0]
            
            for index, X_i in enumerate(X_batch):
                y_i = y_batch[index]
                
                grad_w = 2 * self._weights
                grad_b = 0
                
                check = y_i * (self._weights.dot(X_i) + self._b)
                
                if not check >= 1:
                    grad_w -= self.C * y_i * X_i
                    grad_b -= self.C * y_i
                    
                self._weights -= self.learning_rate * grad_w
                self._b -= self.learning_rate * grad_b
                
            Hinge = np.mean(np.maximum(0, 1 - y * (X.dot(self._weights) + self._b)))
            
            loss = np.linalg.norm(self._weights) ** 2 + self.C * Hinge
                
            self._logging(step, loss, verbose)
    
    def predict(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()
        labels = np.sign(X.dot(self._weights) + self._b)
        labels = ((labels + 1) // 2).astype(int)
        
        return labels
        
    def get_coef(self) -> tuple:
        return self._weights, self._b
    
    def _findBatches(self, X: np.array, y: np.array) -> (np.array, np.array):
        sgd_sample = self.sgd_sample
        if sgd_sample is None:
            return X, y
        elif isinstance(sgd_sample, float):
            sgd_sample = int(X.shape[0] * sgd_sample)
            
        sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample)
        return X[sample_rows_idx], y[sample_rows_idx]
    
    def _logging(self, step: int, grad_error: float, verbose: int) -> None:
        if verbose and step % verbose == 0:
            to_print = step if step != 0 else 'Start'
            print(f"{to_print}, | loss: {grad_error}")