import numpy as np
import pandas as pd
import random

from Methods.TreeReg import MyTreeReg

class MyForestReg():
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16, oob_score: str = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.oob_score = oob_score

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.random_state = random_state

        self._estimators = []
        self.leafs_cnt = 0
        self.fi = {}
        self.oob_score_ = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        random.seed(self.random_state)

        self._precompute_fi(X)

        init_cols = list(X.columns)
        init_rows_cnt, init_cols_cnt = X.shape

        cols_smpl = round(self.max_features * init_cols_cnt)
        rows_smpl = round(self.max_samples * init_rows_cnt)

        if self.oob_score:
            y_val_pred = np.zeros(init_rows_cnt)
            y_val_pred_count = np.zeros(init_rows_cnt)

        for estimator in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl)

            X_sample = X.iloc[rows_idx][cols_idx]
            y_sample = y.iloc[rows_idx]

            mask = ~X.index.isin(rows_idx)
            X_val = X.iloc[mask][cols_idx]

            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, y_sample)
            self._estimators.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            self._add_fi(tree.fi, init_rows_cnt, rows_smpl)

            if self.oob_score:
                y_pred = tree.predict(X_val)
                y_val_pred[mask] += y_pred
                y_val_pred_count[mask] += 1

        if self.oob_score:
            mask = (y_val_pred_count != 0)
            y_val_pred[mask] /= y_val_pred_count[mask]

            self._updateMetric(y[mask],  y_val_pred[mask])

    def predict(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        predictions = np.zeros(n)

        for estimator in self._estimators:
            predictions += estimator.predict(X)

        predictions /= self.n_estimators

        return predictions

    def _precompute_fi(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            self.fi[column] = 0

    def _add_fi(self, tree_fi: dict, rows: int, rows_sample: int) -> None:
        for key, value in tree_fi.items():
            self.fi[key] += value * (rows_sample / rows)

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