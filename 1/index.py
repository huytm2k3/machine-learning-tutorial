import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('data.csv')

# Iterate through each column to check for NaN values
for column in dataset.columns:
    missing_count = dataset[column].isna().sum()  # Count the number of NaN values in the column
    print(f"Column '{column}' has {missing_count} missing values")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X)
X = imputer.transform(X)
print(X)