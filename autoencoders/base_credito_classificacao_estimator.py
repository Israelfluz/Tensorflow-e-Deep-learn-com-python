import pandas as pd
base = pd.read_csv('credit-data.csv')
base.haed()

base.shape

base = base.drop('i#clientid', axis = 1)
base.head()

base.dropna()
base.shape

from sklearn.processing import StandardScaler
scaler_x = StandardScaler()
base[['income', 'age', 'loan']] = scaler_x.fit_transform(base[['income', 'age', 'loan']])
base.haed()

x = base.drop('c#default', axis = 1)
y = base['c#default']

x.haed()              