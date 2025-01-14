from sklearn.preprocessing import StandardScaler

data = [[1.0, 0.0, 0.0, 0.0, 44.0, 72000.0],
 [0.0, 0.0, 1.0, 0.0, 27.0, 48000.0],
 [0.0, 1.0, 0.0, 0.0, 30.0, 54000.0],
 [0.0, 0.0, 1.0, 0.0, 38.0, 61000.0],
 [0.0, 1.0, 0.0, 0.0, 40.0, 63777.77777777778],
 [1.0, 0.0, 0.0, 0.0, 35.0, 58000.0],
 [0.0, 0.0, 1.0, 0.0, 38.77777777777778, 52000.0],
 [1.0, 0.0, 0.0, 0.0, 48.0, 79000.0],
 [0.0, 1.0, 0.0, 0.0, 50.0, 83000.0],
 [0.0, 0.0, 0.0, 1.0, 37.0, 67000.0]]

sc = StandardScaler()
data = sc.fit_transform(data)

print(data)