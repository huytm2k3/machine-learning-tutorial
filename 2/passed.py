

# Importing the necessary libraries

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# from sklearn.preprocessing import LabelEncoder



# Load the dataset

df = pd.read_csv('titanic.csv')



# Identify the categorical data

categorial_features_list = ['Sex','Embarked','Pclass']



# Implement an instance of the ColumnTransformer class

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), categorial_features_list)], remainder = 'passthrough')





# Apply the fit_transform method on the instance of ColumnTransformer

# x = df.iloc[:, :-1].values

x = ct.fit_transform(df)





# Convert the output into a NumPy array

x = np.array(x)



# Use LabelEncoder to encode binary categorical data

le = LabelEncoder()

y = le.fit_transform(df['Survived'])



# Print the updated matrix of features and the dependent variable vector

print("Updated matrix of features: \n", x)

print("Updated dependent variable vector: \n", y)