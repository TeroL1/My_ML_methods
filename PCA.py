import numpy as np
import pandas as pd

class MyPCA():
    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    def __repr__(self) -> str:
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'{__class__.__name__} class: {params}'

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.to_numpy()
        X -= X.mean(axis = 0)
        cov_matrix = np.cov(X, rowvar = False)

        _, W_pca = np.linalg.eigh(cov_matrix)
        W_pca = W_pca[:, -self.n_components:]

        X_new = X.dot(W_pca)
        X_new = pd.DataFrame(X_new)

        return X_new