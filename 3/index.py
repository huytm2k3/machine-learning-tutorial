# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
dataset = pd.read_csv('iris.csv')

# Separate features and target
X = dataset.drop(columns='target').values
y = dataset['target'].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# Print the scaled training and test sets
print(X_train)
print(X_test)