# 7
# import library StandardScler untuk menormalisasi data
from sklearn.preprocessing import StandardScaler

# 6
# Import library LabelEncoder untuk mengubah
# nilai class dari bolean ke matrik/numerik
from sklearn.preprocessing import LabelEncoder
# =================================

# 5
# Import library sklearn untuk mengubah atribut kedalam bentuk matrik
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# =================================

# 4
# Import library sklearn untuk mengubah nilai missing value dengan mean
from sklearn.impute import SimpleImputer
# =================================

# 1
# IMPORT Library numpy dan pandas
import numpy as np
import pandas as pd
# =================================


# 2.Load DATA
# membaca data
dataset = pd.read_csv('data.csv')


# 3.Memisahkan atribut (x) dan kelas(y)
# iloc[baris, kolom]
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)


# 4.Mengubah nilai missing
# mengubah nilai yang hilang dengan most_frequent
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(x[:, 1:4])
x[:, 1:4] = imputer.transform(x[:, 1:4])

# print(x)


# 5.Mengubah atribut(x) nominal menjadi numerik
# mengubah atribut nama kedalam bentuk numerik
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),
                                      [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# print(x)


# 7.Menormalisasi data pada atribut x
# normalisasi data
# sc = StandardScaler()
# x[:, 3:] = sc.fit_transform(x[:, 3:])

print(x)
# =====================================


# 6.Melakukan encoding pada kelas(y)
# mengubah kelas Purchased(yes, no) menjadi matrik(0 1)
# le = LabelEncoder()
# y = le.fit_transform(y)

# print(y)
