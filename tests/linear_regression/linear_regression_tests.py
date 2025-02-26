import pandas as pd

from IBALearn.linear_regression import linear_regression
import numpy as np

np.random.seed(0)

X1 = np.random.normal(loc=0.0, scale=1.0, size=100)

X2 = np.random.normal(loc=0.0, scale=1.0, size=100) >= 0

X3 = X1**3

Y = 30*X1 + 4*X2 + 3*X3 + np.random.normal(0, 0.1, size = 100)
X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3':X3})

lr = linear_regression.LinearRegression(X, Y)
lr.printBetas()

y_hat = lr.predict(X)

print(Y - y_hat)


