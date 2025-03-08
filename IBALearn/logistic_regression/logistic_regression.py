import numpy as np
import pandas as pd
import math
from scipy.linalg import solve_triangular

class LogisticRegression:
    beta = pd.DataFrame
    feature_names = np.array
    X = pd.DataFrame
    X_matrix = np.array(0)
    y = np.array
    classes = np.array
    y_hot_encoding = np.array
    n = int
    p = int
    K = int

    def __init__(self, X: pd.DataFrame, y: np.array):
        self.X = X
        self.y = y

        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))

        self.feature_names = np.array(X.columns)

        self.X_matrix = X.to_numpy()

        self.classes = np.unique(y)

        self.n = self.X_matrix.shape[0]
        self.p = self.X_matrix.shape[1]
        self.K = np.size(self.classes)

        y_hot_temp = np.empty((X.shape[0], np.size(self.classes)))

        for idx, c in zip(range(0, np.size(self.classes)), self.classes):
            y_hot_temp[:, idx] = (y==c).flatten()

        self.y_hot_encoding = y_hot_temp

        beta_temp = np.random.randn(self.p, self.K) * 0.01
        self.beta = pd.DataFrame({str(c): beta_temp[:, idx] for c, idx in zip(self.classes, range(0, np.size(self.classes)))})

        self.__fit_model()

    def __fit_model(self):
        beta = self.beta.to_numpy()
        for _ in range(0, 1000):
            g = self.__computeGradient(self.__getP(beta))
            H = self.__computeHessian(self.__getP(beta))
            H += np.eye(H.shape[0])
            delta_beta = np.linalg.solve(H, g)
            delta_beta = delta_beta.reshape(self.p, self.K)
            beta -= delta_beta

        self.beta = beta

    def __getP(self, beta):
        eta_exp = self.X_matrix @ beta
        eta_exp = np.exp(eta_exp - np.max(eta_exp, axis = 1, keepdims=True))
        return eta_exp / eta_exp.sum(axis=1, keepdims=True)

    def __computeGradient(self, P):
        return (self.X_matrix.T @ (self.y_hot_encoding - P)).flatten()

    def __computeHessian(self, P):
        H_size = self.K * self.p
        H = np.zeros(shape=(H_size, H_size))

        for i in range(self.n):
            P_i = P[i, :].reshape(-1, 1)
            W_i = np.diagflat(P_i) - P_i @ P_i.T

            X_i = self.X_matrix[i, :].reshape(-1,1)
            H_i = np.kron(W_i, X_i @ X_i.T)
            H+=H_i

        return -H

    def predict(self, X):
        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))

        X = X.to_numpy()
        logits = X @ self.beta

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p_mat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return pd.DataFrame({c: p_mat[:, idx] for c, idx in zip(self.classes, range(np.size(self.classes)))})