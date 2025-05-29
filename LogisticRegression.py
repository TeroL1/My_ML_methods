import numpy as np
import pandas as pd
import random
from typing import Union

class MyLogReg():
    """
    Реализация логистической регрессии.
    Поддерживает:
    - различные метрики (accuracy, precision, recall, f1, roc_auc)
    - регуляризации (L1, L2, ElasticNet)
    - стохастический градиентный спуск
    - гибкую настройку обучения
    """
    
    def __init__(self, n_iter: int = 100, learning_rate: Union[callable, float] = 0.1, metric = None, reg = None, l1_coef = 0, l2_coef = 0, sgd_sample: Union[int, float] = None, random_state: int = 42) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
        self.sgd_sample = sgd_sample
        
        self.random_state = random_state
        
        self._weights = None
        self.score = None
        
    def __repr__(self) -> str:
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
            
            y_pred = __class__._sigmoid(X_batch.dot(self._weights))
            logloss_value = __class__._logloss(y_batch, y_pred)
            
            grad_error = 1 / data_len * (y_pred - y_batch).dot(X_batch) + self._computeGradReg()
            
            self._weights -= self._lrCompute(step) * grad_error
            
            self._updateMetric(y, __class__._sigmoid(X.dot(self._weights)))

            self._logging(step, logloss_value, verbose)
            
    def get_coef(self) -> np.array:
        return self._weights[1:]
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()      
        X = np.insert(X, 0, 1, axis=1)
        
        return __class__._sigmoid(X.dot(self._weights))
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.array:
        y = self.predict_proba(X)
        return (y > threshold).astype(int)
    
    def get_best_score(self) -> float:
        try:
            return self.score
        
        except:
            raise AttributeError("The metric is not set.")
            
    def _updateMetric(self, y_true: np.array, y_pred: np.array) -> None:
        if self.metric:
            try:
                self.score = getattr(self, '_' + self.metric)(y_true, y_pred)
            except:
                raise ValueError("There is no such metric.")
    
    @staticmethod
    def _findErrorMatrix(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> (int, int, int, int):
        TP = len(y_true[(y_pred >= threshold) & (y_true == 1)])
        FN = len(y_true[(y_pred < threshold) & (y_true == 1)])
        FP = len(y_true[(y_pred >= threshold) & (y_true == 0)])
        TN = len(y_true[(y_pred < threshold) & (y_true == 0)])
        
        return TP, FN, FP, TN
    
    @staticmethod
    def _accuracy(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
        TP, FN, FP, TN = __class__._findErrorMatrix(y_true, y_pred, threshold)
        accuracy_value = (TP + TN) / (TP + FN + FP + TN) if (TP + FN + FP + TN) > 0 else 0
        
        return accuracy_value
    
    @staticmethod
    def _precision(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
        TP, FN, FP, TN = __class__._findErrorMatrix(y_true, y_pred, threshold)
        precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        return precision_value
    
    @staticmethod
    def _recall(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
        TP, FN, FP, TN = __class__._findErrorMatrix(y_true, y_pred, threshold)
        recall_value = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        return recall_value
    
    @staticmethod
    def _f1(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
        precision_value = __class__._precision(y_true, y_pred, threshold)
        recall_value = __class__._recall(y_true, y_pred, threshold)
        
        f1_value = 2 * (precision_value * recall_value) / (precision_value + recall_value) if precision_value + recall_value > 0 else 0
        return f1_value
    
    @staticmethod
    def _roc_auc(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> float:
        TP, FN, FP, TN = __class__._findErrorMatrix(y_true, y_pred, threshold)
        P = TP + FN
        F = FP + TN
        
        if P == 0 or F == 0:
            return 0
        
        idx = np.argsort(-y_pred)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = len(y_pred)
        
        j = 0
        ones_global = 0
        roc_auc = 0
        ones_here = 0
        for i in range(0, n):
            while j < n and y_pred[j] == y_pred[i]:
                if y_true[j] == 1:
                    ones_here += 1
                j += 1
               
            if y_true[i] == 0:
                roc_auc += ones_here / 2 + ones_global
            
            if i + 1 < n and y_pred[i] != y_pred[i + 1]:
                ones_global += ones_here
                ones_here = 0
                
        roc_auc /= (P * F)
        
        return roc_auc       
    
    @staticmethod
    def _sigmoid(value: np.array) -> np.array:
        return 1 / (1 + np.exp(-value))
    
    @staticmethod
    def _logloss(y_true: np.array, y_pred: np.array, eps: float = 10 ** (-15)) -> np.array:
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    
    def _lrCompute(self, step: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(step)
        
        else:
            return self.learning_rate
    
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