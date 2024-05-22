from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = [
    ['France', 44.0, 72000.0],
    ['Spain', 27.0, 48000.0],
    ['Germany', 30.0, 54000.0],
    ['Spain', 38.0, 61000.0],
    ['Germany', 40.0, 63777.77777777778],
    ['France', 35.0, 58000.0],
    ['Spain', 38.77777777777778, 52000.0],
    ['France', 48.0, 79000.0],
    ['Germany', 50.0, 83000.0],
    ['Vietnam', 37.0, 67000.0],
]


X = np.array(ct.fit_transform(X))
print(X)