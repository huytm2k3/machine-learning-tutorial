# Bước 1: Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Bước 2: Tải dataset Titanic vào một DataFrame của pandas
dataset = pd.read_csv('titanic.csv')

# Bước 3: Xác định các đặc trưng phân loại trong dataset cần được mã hóa
# Dựa vào dữ liệu, các cột phân loại thường là Sex', 'Embarked', 'Pclass'
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Bước 4: Tạo một instance của lớp ColumnTransformer để áp dụng OneHotEncoder
# Lưu ý: Cần truyền vào danh sách chỉ số cột hoặc tên cột phân loại
ct = ColumnTransformer(
    transformers=[(
        'encoder',
        OneHotEncoder(),
        categorical_features
    )],
    remainder='passthrough'
)

# Bước 5: Sử dụng phương thức fit_transform trên instance của ColumnTransformer để áp dụng OneHotEncoding
# Tách đặc trưng (X) và nhãn (y)
y = dataset['Survived']
X = dataset

# Áp dụng ColumnTransformer 
# để mã hóa OneHot các đặc trưng phân loại
X = ct.fit_transform(X)

# Bước 6: Chuyển đổi đầu ra của fit_transform thành mảng NumPy để sử dụng thêm
X = np.array(X)

# Bước 7: Mã hóa cột 'Survived' (biến phụ thuộc) bằng LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Bước 8: In ma trận đặc trưng đã cập nhật và vector biến phụ thuộc
print(X.shape)
print(X)
print(y)