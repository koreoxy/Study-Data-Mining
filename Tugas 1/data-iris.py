from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['Species'])

# print(iris.feature_names)
# print(iris.target_names)

print(iris.target.shape)
print(iris.data)

# print(df['Species'].value_counts())
print(df.describe())
