import pandas as pd
import numpy as np
import scipy as sci

class BinaryLogisticRegression:
    X = np.array
    y = np.array
    beta = np.array

    features = np.array
    classes = np.array

    n = int
    p = int
    K = int

    def __getP(self, predX):
        ext = (predX @ self.beta)
        return 1 / (1 + np.exp(-np.clip(ext, -500, 500)))

    def __getg(self):
        return self.X.T @ (self.y - self.__getP(self.X))

    def __getW(self):
        P = self.__getP(self.X)
        epsilon = 1e-6  # Small term to prevent singularity
        W_diag = np.diag(P * (1 - P) + epsilon)

        return W_diag

    def __getHessian(self):
        return - (self.X.T @ self.__getW() @ self.X)

    def updateStep(self):
        g = self.__getg()
        W = self.__getW()
        A = np.sqrt(W) @ self.X

        Q, R = np.linalg.qr(A)

        try:
            z = sci.linalg.solve_triangular(R.T, -g, lower=True)
            sigma = sci.linalg.solve_triangular(R, z)

            # Scale updates to prevent divergence
            step_size = min(1.0, 1.0 / np.linalg.norm(sigma))
            self.beta = self.beta + step_size * sigma
        except np.linalg.LinAlgError:
            print("Singular matrix encountered in update step, skipping update.")

    def __init__(self, X : pd.DataFrame, y : np.array, true_case):
        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))
        self.X = X.to_numpy(dtype=np.float64)
        self.y = y == true_case

        np.random.seed(0)
        self.beta = np.random.normal(0, 0.0001, size=X.shape[1])

        self.features = X.columns
        self.classes = np.unique(y)

        self.n, self.p = X.shape
        self.K = np.size(self.classes)

        self.fit_model()

    def fit_model(self):
        prev_beta = self.beta
        update = 0
        while True:
            self.updateStep()
            print(np.sum(prev_beta - self.beta))
            update = update + 1
            if abs(np.sum(prev_beta - self.beta)) <= 1e-6 or update == 1000:
                break
            else:
                prev_beta = self.beta

    def predictX(self, X : pd.DataFrame):
        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))

        return self.__getP(X.to_numpy())
