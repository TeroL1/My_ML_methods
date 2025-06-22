import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature: str = None, threshold: float = None, left: 'TreeNode' = None, right: 'TreeNode' = None, depth: int = 0) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth

        self.is_leaf = False
        self.value = None

    def make_leaf(self, y: pd.Series) -> None:
        self.is_leaf = True
        self.value = np.mean(y)

class MyTreeReg():
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.fi = {}
        self.X_len = None

        self._root = None
        self.leafs_cnt = 0
        self.potential_leaves = 1
        self._bins_dict = {}

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key[0] != '_')
        return f'{__class__.__name__} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._precompute_fi(X)
        self.X_len = X.shape[0]

        if self.bins:
            self._build_bins_dict(X)

        self._root = self.rec_build_tree(X, y)

    def rec_build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> TreeNode:
        if (depth == self.max_depth) or (X.shape[0] < self.min_samples_split) or (self.leafs_cnt + self.potential_leaves >= self.max_leafs and depth >= 1) or (len(np.unique(y)) == 1):

            return self._make_leaf(y, depth)

        col_name, split_value, ig = self.get_best_split(X, y)

        self.fi[col_name] += (X.shape[0] / self.X_len) * ig

        X_left = X.loc[X[col_name] <= split_value]
        y_left = y.loc[X[col_name] <= split_value]

        X_right = X.loc[X[col_name] > split_value]
        y_right = y.loc[X[col_name] > split_value]

        self.potential_leaves += 1

        left_value = self.rec_build_tree(X_left, y_left, depth + 1)
        right_value = self.rec_build_tree(X_right, y_right, depth + 1)

        node = TreeNode(col_name, split_value, left_value, right_value, depth)

        return node

    def predict(self, X: pd.DataFrame) -> np.array:
        n = X.shape[0]
        predictions = np.zeros(n)

        for index in range(n):
            row = X.iloc[index]
            predictions[index] = __class__._rec_predict(row, self._root)

        return predictions

    def print_tree(self) -> None:
        __class__._print_tree_rec(self._root)

    def sum_of_leafs(self) -> float:
        return __class__._rec_sum_of_leafs(self._root)

    def _build_bins_dict(self, X: pd.DataFrame) -> None:
        for name, column in X.items():
            thresholds = np.unique(sorted(column))
            if len(thresholds) > self.bins - 1:
                thresholds = np.histogram(column, self.bins)[1][1:-1]
                self._bins_dict[name] = thresholds

            else:
                self._bins_dict[name] = thresholds

    def _get_thresholds(self, column: pd.Series, name: str) -> np.array:
        if self.bins:
            return self._bins_dict[name]

        thresholds = np.unique(sorted(column))
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        return thresholds

    def _make_leaf(self, y: pd.Series, depth: int) -> TreeNode:
        leaf = TreeNode(depth = depth)
        leaf.make_leaf(y)

        self.leafs_cnt += 1
        self.potential_leaves -= 1

        return leaf

    def _precompute_fi(self, X: pd.DataFrame) -> None:
        for column in X.columns:
            self.fi[column] = 0

    @staticmethod
    def _rec_predict(vector: pd.Series, node: TreeNode) -> float:
        if node.is_leaf == True:
            return node.value

        elif vector[node.feature] > node.threshold:
            return __class__._rec_predict(vector, node.right)

        return __class__._rec_predict(vector, node.left)
    @staticmethod
    def _rec_sum_of_leafs(node: TreeNode) -> float:
        if node.is_leaf == True:
            return node.value

        leafs_sum = 0

        left = __class__._rec_sum_of_leafs(node.left)
        right = __class__._rec_sum_of_leafs(node.right)

        leafs_sum += (left + right)

        return leafs_sum

    @staticmethod
    def _print_tree_rec(node: TreeNode, side: str = None, depth: int = 0) -> None:
        if (node.is_leaf == True):
            print(f"{' ' * 3 * depth}leaf_{side} = {node.value}")

        else:
            print(f"{' ' * 3 * depth}{node.feature} > {node.threshold}")
            __class__._print_tree_rec(node.left, 'left', depth + 1)
            __class__._print_tree_rec(node.right, 'right', depth + 1)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        col_name, split_value, best_ig = None, None, float('-inf')

        s_0 = __class__.get_entropy(y)
        for feature in X.columns:
            column = X[feature]
            current_ig, current_split_value = self.get_best_column_split(column, y, feature, s_0)

            if current_ig > best_ig:
                col_name, split_value, best_ig = feature, current_split_value, current_ig

        return col_name, split_value, best_ig

    def get_best_column_split(self, column: pd.Series, y: pd.Series, name: str, s_0: float) -> tuple:
        thresholds = self._get_thresholds(column, name)

        best_ig, split_value = float('-inf'), None

        for threshold in thresholds:
            ig = __class__.get_column_split(column, y, threshold, s_0)

            if ig > best_ig:
                best_ig, split_value = ig, threshold

        return best_ig, split_value

    @staticmethod
    def get_column_split(column: pd.Series, y: pd.Series, threshold: float, s_0: float) -> float:
        y_1 = y[column <= threshold]
        y_2 = y[column > threshold]

        s_1 = __class__.get_entropy(y_1)
        s_2 = __class__.get_entropy(y_2)

        return s_0 - (s_1 * len(y_1) / len(y) + s_2 * len(y_2) / len(y))

    @staticmethod
    def get_entropy(y: pd.Series) -> float:
        y_mean = np.mean(y)

        return np.mean((y - y_mean) ** 2)

    @property
    def root(self):
        return self._root
