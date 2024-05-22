# Import necessary libraries
import pandas as pd
import numpy as np

# Load the Wine Quality Red dataset
dataset = pd.read_csv('winequality-red.csv', sep=';')

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Create an instance of the StandardScaler class
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
sc.fit(X_train)
X_train = sc.transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print(X_train)
print(X_test)