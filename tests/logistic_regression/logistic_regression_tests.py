import pandas as pd

from IBALearn.logistic_regression import logistic_regression
import numpy as np

np.random.seed(0)

X1 = np.random.normal(scale = 1, size = 300)

YBool = X1 + X1**3 + np.random.normal(scale = 2, size = 300) > 1

Y = np.full((300,), "False")

Y[YBool] = "True"

x = pd.DataFrame({"X1": X1})

mod = logistic_regression.LogisticRegression(x, Y)

X2 = np.linspace(0.5, 1.5, 20)

x2 = pd.DataFrame({"X2": X2})
print(mod.predict(x2).to_numpy())
