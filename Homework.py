import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('Bestsellers with categories.csv')

print(df.isnull().sum())

df = df.fillna(method='ffill')

print(df['Genre'].value_counts())

df['Genre'] = df['Genre'].astype('category').cat.codes
print(df['Genre'].median())

df['Genre'].value_counts().plot.bar()
plt.show()

df['Genre'].value_counts().plot.pie()
plt.show()

num_df = df.select_dtypes(include=['int64', 'float64'])
num_df.plot.box()
plt.show()

standard = StandardScaler().fit_transform(num_df)
normalized = MinMaxScaler().fit_transform(num_df)