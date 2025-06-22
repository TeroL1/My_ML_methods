import numpy as np
import pandas as pd
import random

from Methods.TreeClf import MyTreeClf

class MyForestClf():
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16, criterion: str = 'entropy', oob_score: str = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.oob_score = oob_score

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion

        self.random_state = random_state

        self._estimators = []
        self.leafs_cnt = 0
        self.fi = {}
        self.oob_score_ = None

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'


    def fit(self, X, y) -> None:
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

            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, self.criterion)
            tree.fit(X_sample, y_sample)
            self._estimators.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            self._add_fi(tree.fi, init_rows_cnt, rows_smpl)

            if self.oob_score:
                y_pred = tree.predict_proba(X_val)
                y_val_pred[mask] += y_pred
                y_val_pred_count[mask] += 1

        if self.oob_score:
            mask = (y_val_pred_count != 0)
            y_val_pred[mask] /= y_val_pred_count[mask]

            self._updateMetric(y[mask].to_numpy(),  y_val_pred[mask])

    def predict(self, X: pd.DataFrame, type: str) -> np.array:
        return getattr(self, '_' + type + '_predict')(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        predictions = np.zeros(n)

        for estimator in self._estimators:
            predictions += estimator.predict_proba(X)

        predictions /= self.n_estimators

        return predictions

    def _mean_predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.array:
        predictions = self.predict_proba(X)
        predictions = (predictions > threshold)

        return predictions.astype(int)

    def _vote_predict(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        predictions = np.zeros(n)

        for estimator in self._estimators:
            predictions += estimator.predict(X)

        predictions /= self.n_estimators
        predictions = (predictions > 0.5)

        return predictions.astype(int)

    def _precompute_fi(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            self.fi[column] = 0

    def _add_fi(self, tree_fi: dict, rows: int, rows_sample: int) -> None:
        for key, value in tree_fi.items():
            self.fi[key] += value * (rows_sample / rows)

    def _updateMetric(self, y_true: np.array, y_pred: np.array) -> None:
        if self.oob_score:
            try:
                self.oob_score_ = getattr(self, '_' + self.oob_score)(y_true, y_pred)
            except:
                raise ValueError("There is no such metric.")

    @staticmethod
    def _findErrorMatrix(y_true: np.array, y_pred: np.array, threshold: float = 0.5) -> tuple:
        TP = len(y_true[(y_pred > threshold) & (y_true == 1)])
        FN = len(y_true[(y_pred <= threshold) & (y_true == 1)])
        FP = len(y_true[(y_pred > threshold) & (y_true == 0)])
        TN = len(y_true[(y_pred <= threshold) & (y_true == 0)])

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
        y_pred = np.round(y_pred, 10)
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