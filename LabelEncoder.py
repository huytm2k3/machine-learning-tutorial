from sklearn.preprocessing import LabelEncoder
import random

y = []

choices = ['Phuong', 'Huy']
for i in range (0, 10):
    y.append(random.choice(choices))

print(y)

le = LabelEncoder()
y = le.fit_transform(y)

print(y)