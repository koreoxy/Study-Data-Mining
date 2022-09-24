# LOAD ATAU MEMASUKAN LIBRARY SKLEARN
from sklearn.datasets import load_wine


# import pandas dan numpy
import pandas as pd
import numpy as np


# Variable
wine = load_wine()


# variable
x = wine.data
y = wine.target

# variable
df = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['Species'])


print(df.describe())

# Print hasil
# print(df['Species'].value_counts())

# print untuk melihat hasil dari variable wine tersebut
# print(wine.feature_names)

# untuk melihat class
# print(wine.target_names)

# untuk melihat jumlah data
# print(wine.target.shape)


# print features data wine
# print(wine.data)
