import numpy as np
import pandas as pd
from typing import Union
import random

from Methods.TreeReg import TreeNode
from Methods.TreeReg import MyTreeReg

class MyBoostClf():
    def __init__(self, n_estimators: int = 10, learning_rate: Union[callable, float] = 0.1, metric: str = None, reg: float = 0.1, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16, max_features: float = 0.5, max_samples: float = 0.5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.trees = []
        self.pred_0 = None
        self.best_score = None
        self.leafs_cnt = 0
        self.fi = {}
        self.not_change_score = 0

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, early_stopping: int = None, verbose: int = None) -> None:
        random.seed(self.random_state)

        self._precompute_fi(X)

        self.pred_0 = __class__._computeLogOdds(y)
        Fm = np.full_like(y.to_numpy(), self.pred_0, dtype=np.float64)
        Fm = pd.Series(Fm)

        init_cols = list(X.columns)
        init_rows_cnt, init_cols_cnt = X.shape

        cols_smpl = round(self.max_features * init_cols_cnt)
        rows_smpl = round(self.max_samples * init_rows_cnt)

        for epoch in range(1, self.n_estimators + 1):
            cols_idx = random.sample(init_cols, cols_smpl)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl)

            X_sample = X.iloc[rows_idx][cols_idx]
            y_sample = y.iloc[rows_idx]
            Fm_sample = Fm.iloc[rows_idx]

            error = pd.Series(__class__._computeAntiGrad(y_sample, Fm_sample))
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, error)

            self._RecChange(X_sample, y_sample, Fm_sample, tree.root)
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            lr = self._lrCompute(epoch)
            Fm += lr * tree.predict(X)

            Fm_valid = Fm if X_val is None else self._compute_Fm(X_val)
            y_valid = y if y_val is None else y_val

            logging_loss = __class__._computeLoss(y, Fm)
            y_pred = __class__._findP(Fm_valid)
            metric = self._updateMetric(y_valid.to_numpy(), y_pred)
            score = metric if metric else self._computeLoss(y_valid, Fm_valid)
            self._early_stopping_count(self.best_score, score)

            self._add_fi(tree.fi, init_rows_cnt, rows_smpl)
            self._logging(epoch, logging_loss, verbose)

            if self._early_stopping(early_stopping):
                break

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        prediction = self._compute_Fm(X)
        prediction = __class__._findP(prediction)

        return prediction

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.array:
        prediction = self.predict_proba(X)
        prediction = (prediction > threshold)

        return prediction.astype(int)

    def _compute_Fm(self, X: pd.DataFrame) -> np.array:
        Fm = self.pred_0

        for epoch, tree in enumerate(self.trees, 1):
            lr = self._lrCompute(epoch)
            Fm += lr * tree.predict(X)

        return Fm

    def _lrCompute(self, step: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(step)
        else:
            return self.learning_rate

    @staticmethod
    def _findP(y: np.array) -> np.array:
        p = np.exp(y) / (1 + np.exp(y))
        return p

    @staticmethod
    def _computeAntiGrad(y_true: np.array, y_pred: np.array) -> float:
        p = __class__._findP(y_pred)
        Grad = y_true - p
        return Grad

    @staticmethod
    def _computeLogOdds(y: np.array) -> float:
        eps = 10 ** (-15)
        p = y.mean()
        LogOdds = np.log((p + eps)/ (1 - p + eps))
        return LogOdds

    @staticmethod
    def _findGamma(y: np.array, p: np.array) -> float:
        gamma = np.sum(y - p) / np.sum(p * (1 - p))
        return gamma

    @staticmethod
    def _computeLoss(y: np.array, LogOdds: np.array) -> float:
        loss = - (y * LogOdds - np.log(1 + np.exp(LogOdds))).mean()
        return loss

    def _updateMetric(self, y_true: np.array, y_pred: np.array) -> float:
        if self.metric:
            try:
                return getattr(self, '_' + self.metric)(y_true, y_pred)
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

    def _precompute_fi(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            self.fi[column] = 0.0

    def _add_fi(self, tree_fi: dict, rows: int, rows_sample: int) -> None:
        for key, value in tree_fi.items():
            self.fi[key] += value * (rows_sample / rows)

    def _RecChange(self, X: pd.DataFrame, y: pd.Series, Fm: np.ndarray, node: TreeNode) -> None:
        if node is None:
            return

        if node.is_leaf:
            indices = X.index
            y_, p_ = y.loc[indices], __class__._findP(Fm[indices])
            node.value = self._findGamma(y_, p_) + self.reg * self.leafs_cnt
            return

        mask_right = X[node.feature] > node.threshold
        mask_left = ~mask_right

        right_idx = X[mask_right].index
        left_idx = X[mask_left].index

        self._RecChange(X.loc[right_idx], y, Fm, node.right)
        self._RecChange(X.loc[left_idx], y, Fm, node.left)

    def _early_stopping_count(self, best_value: int, value: int) -> None:
        if self.best_score is None:
            self.best_score = value
            self.not_change_score = 0
            return

        if value > best_value:
            if not self.metric:
                self.not_change_score += 1

            else:
                self.best_score = value

        elif value < best_value:
            if self.metric:
                self.not_change_score += 1

            else:
                self.best_score = value

    def _early_stopping(self, count) -> bool:
        if self.not_change_score == count:
            for _ in range(count):
                self.trees.pop()

            return True

        return False

    def _logging(self, step: int, loss: float, verbose: int) -> None:
        if verbose and step % verbose == 0:
            to_print = f"{step}. Loss[{self.loss}]: {loss}"
            if self.metric:
                to_print += f" | {self.metric}: {self.best_score}"

            print(to_print)