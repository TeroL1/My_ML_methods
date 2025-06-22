import numpy as np
import pandas as pd
import random
import copy

class MyBaggingReg():
    def __init__(self, estimator = None, n_estimators: int = 10, max_samples: float = 1.0, oob_score: str = None, random_state: int = 42) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.oob_score = oob_score

        self.random_state = random_state

        self.estimators = []
        self.oob_score_ = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        random.seed(self.random_state)

        init_rows_cnt = X.shape[0]
        rows_smpl_cnt = round(init_rows_cnt * self.max_samples)

        if self.oob_score:
            y_val_pred = np.zeros(init_rows_cnt)
            y_val_pred_count = np.zeros(init_rows_cnt)

        samples = []

        for _ in range(self.n_estimators):
            index = random.choices(list(range(init_rows_cnt)), k = rows_smpl_cnt)
            samples.append(index)

        for index in samples:
            X_sample = X.iloc[index]
            y_sample = y.iloc[index]

            mask = ~X.index.isin(index)
            X_val = X.iloc[mask]

            estimator = copy.copy(self.estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

            if self.oob_score:
                y_pred = estimator.predict(X_val)
                y_val_pred[mask] += y_pred
                y_val_pred_count[mask] += 1

        if self.oob_score:
            mask = (y_val_pred_count != 0)
            y_val_pred[mask] /= y_val_pred_count[mask]

            self._updateMetric(y[mask],  y_val_pred[mask])

    def predict(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        predictions = np.zeros(n)

        for estimator in self.estimators:
            predictions += estimator.predict(X)

        predictions /= self.n_estimators

        return predictions

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
        if self.oob_score:
            try:
                self.oob_score_ = getattr(self, '_' + self.oob_score)(y_true, y_pred)

            except:
                raise ValueError("There is no such metric.")