import numpy as np
import pandas as pd
import random
from typing import Union

class MyLineReg():
    """
    Реализация линейной регрессии.
    Поддерживает:
    - различные метрики (MSE, MAE, RMSE, MAPE, R2)
    - регуляризации (L1, L2, ElasticNet)
    - стохастический градиентный спуск
    - гибкую настройку обучения
    """
    
    def __init__(self, n_iter: int = 100, learning_rate: Union[callable, float] = 0.1, metric: str = None, reg = None, l1_coef = 0, l2_coef = 0, sgd_sample: Union[int, float] = None, random_state: int = 42) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
        self.sgd_sample = sgd_sample
        
        self.random_state = random_state
        
        self._weights = None
        self._score = None
        
    def __repr__(self):
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'{__class__.__name__} class: {params}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        random.seed(self.random_state)
        
        _, features_len = X.shape[0], X.shape[1] + 1
        X = X.to_numpy()
        y = y.to_numpy()
        
        X = np.insert(X, 0, 1, axis=1)
        
        self._weights = np.ones(features_len)
        
        for step in range(1, self.n_iter + 1):
            X_batch, y_batch = self._findBatches(X, y)
            data_len = X_batch.shape[0]
            
            y_pred = X_batch.dot(self._weights)

            MSE = 2 / data_len * (y_pred - y_batch).dot(X_batch)
            
            grad_error = MSE + self._computeGradReg()
            
            self._weights -= self._lrCompute(step) * grad_error
            
            self._updateMetric(y, X.dot(self._weights))
                    
            self._logging(step, np.mean(MSE), verbose)
                
    def get_best_score(self) -> float:
        try:
            return self.score
        
        except:
            raise AttributeError("The metric is not set.")
    
    def get_coef(self) -> np.array:
        return self._weights[1:]

    def predict(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()      
        X = np.insert(X, 0, 1, axis=1)
        
        return X.dot(self._weights)
    
    def _lrCompute(self, step: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(step)
        
        else:
            return self.learning_rate
        
    @staticmethod
    def _mae(y_true: np.array, y_pred: np.array) -> float:
        return (np.abs(y_true - y_pred)).mean()

    @staticmethod
    def _mse(y_true: np.array, y_pred: np.array)  -> float:
        return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def _rmse(y_true: np.array, y_pred: np.array) -> float:
        return np.sqrt(((y_true - y_pred)**2).mean())

    @staticmethod
    def _mape(y_true: np.array, y_pred: np.array) -> float:
        return 100 * (np.abs((y_true - y_pred) / y_true)).mean()

    @staticmethod
    def _r2(y_true: np.array, y_pred: np.array) -> float:
        return 1 - ((y_true - y_pred) ** 2).sum() / (((y_true - y_true.mean()) ** 2).sum())
    
    def _updateMetric(self, y_true: np.array, y_pred: np.array) -> None:
        if self.metric:
            try:
                self.score = getattr(self, '_' + self.metric)(y_true, y_pred)
            except:
                raise ValueError("There is no such metric.")
     
    def _l1Grad(self) -> np.array:
        return self.l1_coef * np.sign(self._weights)
    
    def _l2Grad(self) -> np.array:
        return self.l2_coef * 2 * self._weights
    
    def _elasticnetGrad(self) -> np.array:
        return self._l1Grad() + self._l2Grad()
    
    def _computeGradReg(self) -> np.array:
        if self.reg:
            try:
                return getattr(self, '_' + self.reg + 'Grad')()
            except:
                raise ValueError("There is no such reg.")
                
        return self._weights * 0
    
    def _findBatches(self, X: np.array, y: np.array) -> (np.array, np.array):
        sgd_sample = self.sgd_sample
        if sgd_sample is None:
            return X, y
        elif isinstance(sgd_sample, float):
            sgd_sample = int(X.shape[0] * sgd_sample)
            
        sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample)
        return X[sample_rows_idx], y[sample_rows_idx]
    
    def _logging(self, step: int, grad_error: float, verbose: int) -> None:
        if verbose and (step - 1) % verbose == 0:
            to_print = step if step != 1 else 'Start'
            if self.metric:
                print(f"{to_print}, | loss: {grad_error}, |, self.metric: {self.score}")
            else:
                print(f"{to_print}, | loss: {grad_error}")