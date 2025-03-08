import numpy as np
import pandas as pd

from IBALearn.logistic_regression_binary import BinaryLogisticRegression
np.random.seed(0)

X = np.random.normal(0, 2, 100)
Xpd = pd.DataFrame({"X": X})
yBool = X + np.random.normal(0, 2, 100) >= 0

y = np.full((100,), "False")

y[yBool] = "True"

mod = BinaryLogisticRegression(Xpd, y, "True")

Xpred = np.linspace(-1, 1, 10)

Xpred = pd.DataFrame({"X": Xpred})
print(mod.predictX(Xpred))



