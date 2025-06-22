import numpy as np
import pandas as pd
from typing import Union
import random

from Methods.TreeReg import TreeNode
from Methods.TreeReg import MyTreeReg

class MyBoostReg():
    def __init__(self, n_estimators: int = 10, learning_rate: Union[callable, float] = 0.1, loss: str = 'MSE', metric: str = None, reg: float = 0.1, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16, max_features: float = 0.5, max_samples: float = 0.5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.metric = metric
        self.reg = reg

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.pred_0 = None
        self.trees = []
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

        self.pred_0 = self._computeLossVal(y)
        Fm = np.full_like(y.to_numpy(), self.pred_0)
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

            error = - self._computeGrad(y_sample, Fm_sample)
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_sample, error)

            self._RecChange(X_sample, y_sample, Fm_sample, tree.root)
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            lr = self._lrCompute(epoch)
            Fm += lr * tree.predict(X)

            Fm_valid = Fm if X_val is None else self._compute_Fm(X_val)
            y_valid = y if y_val is None else y_val

            logging_loss = self._computeLoss(y, Fm)
            metric = self._updateMetric(y_valid, Fm_valid)
            score = metric if metric else self._computeLoss(y_valid, Fm_valid)
            self._early_stopping_count(self.best_score, score)

            self._add_fi(tree.fi, init_rows_cnt, rows_smpl)
            self._logging(epoch, logging_loss, verbose)

            if self._early_stopping(early_stopping):
                break

    def _compute_Fm(self, X: pd.DataFrame) -> np.array:
        Fm = self.pred_0

        for epoch, tree in enumerate(self.trees, 1):
            lr = self._lrCompute(epoch)
            Fm += lr * tree.predict(X)

        return Fm

    def predict(self, X: pd.DataFrame) -> np.array:
        return self._compute_Fm(X)

    def _lrCompute(self, step: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(step)

        else:
            return self.learning_rate

    def _computeGrad(self, y_true: np.array, y_pred: np.array) -> float:
        return getattr(self, '_' + self.loss + 'Grad')(y_true, y_pred)

    def _computeLossVal(self, y: np.array) -> float:
        return getattr(self, '_' + self.loss + 'Val')(y)

    @staticmethod
    def _MSEGrad(y_true: np.array, y_pred: np.array) -> float:
        return 2 * (y_true - y_pred)

    @staticmethod
    def _MAEGrad(y_true: np.array, y_pred: np.array) -> float:
        return np.sign(y_pred - y_true)

    @staticmethod
    def _MSEVal(y: np.array) -> float:
        return y.mean()

    @staticmethod
    def _MAEVal(y: np.array) -> float:
        return y.median()

    @staticmethod
    def _MSE(y_true: np.array, y_pred: np.array)  -> float:
        return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def _MAE(y_true: np.array, y_pred: np.array) -> float:
        return (np.abs(y_true - y_pred)).mean()

    @staticmethod
    def _RMSE(y_true: np.array, y_pred: np.array) -> float:
        return np.sqrt(((y_true - y_pred)**2).mean())

    @staticmethod
    def _MAPE(y_true: np.array, y_pred: np.array) -> float:
        return 100 * (np.abs((y_true - y_pred) / y_true)).mean()

    @staticmethod
    def _R2(y_true: np.array, y_pred: np.array) -> float:
        return 1 - ((y_true - y_pred) ** 2).sum() / (((y_true - y_true.mean()) ** 2).sum())

    def _computeLoss(self, y_true: np.array, y_pred: np.array) -> float:
        if self.loss == 'MSE' or self.loss == 'MAE':
            return getattr(self, '_' + self.loss)(y_true, y_pred)

    def _updateMetric(self, y_true: np.array, y_pred: np.array) -> float:
        if self.metric:
            try:
                return getattr(self, '_' + self.metric)(y_true, y_pred)
            except:
                raise ValueError("There is no such metric.")

    def _DFSchange(self, y: np.array, node: TreeNode) -> None:
        if node:
            if node.is_leaf:
                node.value = self._computelossVal(y - node.value)

            self._DFSchange(y, node.left)
            self._DFSchange(y, node.right)

    def _RecChange(self, X: pd.DataFrame, y: pd.Series, Fm: np.ndarray, node: TreeNode) -> None:
        if node is None:
            return

        if node.is_leaf:
            indices = X.index
            error = y.loc[indices] - Fm[indices]
            node.value = self._computeLossVal(error) + self.reg * self.leafs_cnt
            return

        mask_right = X[node.feature] > node.threshold
        mask_left = ~mask_right

        right_idx = X[mask_right].index
        left_idx = X[mask_left].index

        self._RecChange(X.loc[right_idx], y.loc[right_idx], Fm, node.right)
        self._RecChange(X.loc[left_idx], y.loc[left_idx], Fm, node.left)

    def _precompute_fi(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            self.fi[column] = 0.0

    def _add_fi(self, tree_fi: dict, rows: int, rows_sample: int) -> None:
        for key, value in tree_fi.items():
            self.fi[key] += value * (rows_sample / rows)

    def _early_stopping_count(self, best_value: int, value: int) -> None:
        if self.best_score is None:
            self.best_score = value
            self.not_change_score = 0
            return

        if value > best_value:
            if self.metric != 'R2':
                self.not_change_score += 1

            else:
                self.best_score = value

        elif value < best_value:
            if self.metric == 'R2':
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