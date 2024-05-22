from sklearn.impute import SimpleImputer

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [None, 11, 5],
    [13, 14, 15],
    [16, 17, None],
    [19, 20, 21]
]

imp = SimpleImputer(strategy='mean')
imp.fit(data)

transformed_data = imp.transform(data)
print(transformed_data)