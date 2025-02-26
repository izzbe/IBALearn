import numpy as np
import pandas as pd
import math
from scipy.linalg import solve_triangular

class LinearRegression:
    beta = np.array 
    feature_names = np.array
    X = pd.DataFrame
    X_matrix = np.array(0)
    y = np.array
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.X = X
        self.y = y
        
        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))

        self.feature_names = np.array(X.columns)

        X_matrix = X.to_numpy() 
        self.X_matrix = X_matrix

        Q, R =  np.linalg.qr(X_matrix)
        self.beta = solve_triangular(R, np.matrix.transpose(Q) @ y)

    def __getRSS(self):
        RSS = 0
        for y_hat, idx in zip(self.predict(self.X), range(0, self.y.size)):
            RSS += (y_hat - self.y[idx])**2
        
        return RSS

    def __getMSE(self):
        MSE = self.__getRSS() * (1 / (self.X.shape[1] - self.X.shape[0]))
        return MSE
    
    def predict(self, X:pd.DataFrame):
        X = pd.get_dummies(X)
        X.insert(0, 'intercept', np.ones(X.shape[0]))
        return X @ self.beta
    
    def getZScore(self):
        ZScores = []
        Q, R = np.linalg.qr(self.X_matrix)

        sqrt_MSE = math.sqrt(self.__getMSE())

        for beta, idx in zip(self.beta, range(0, self.beta.size)):
            e_j = np.zeros(R.shape[1])
            e_j[idx] = 1

            R_T_inv_j = solve_triangular(R.T, e_j, lower = True)

            v_j = R_T_inv_j @ R_T_inv_j

            ZScores.append(beta / (sqrt_MSE * v_j))
        
        return ZScores

    def printBetas(self):
        for col, beta in zip(self.feature_names, self.beta):
            print(col + ': ')
            print(f"{beta:.2f}")
            print("\n")

