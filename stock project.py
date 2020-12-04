import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#import data
data= pd.read_excel("C:/Users/User/Desktop/Source Code/Python/Machine Learning/stock.xlsx")

#cek data dan kolom
print(data)
print(data.columns)

#summary statistic
print(data.describe())

#variance and std deviation
print(data['Harga'].var())
print(data['Harga'].std())

## build a DataFrame
print(data.shape)

X = data[['ROE','ROA','DER','PBV','Aset']]
Y = data['Harga']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#pilih model regresi linear
model = LinearRegression()

#fitting model
model.fit(X_train, Y_train)
print (model.fit)

#eksekusi model
model.intercept_.round(2)
model.coef_.round(2)
print(model.intercept_.round(2))    
print(model.coef_.round(2))

#show r square & MSE
model.score(X_test, Y_test)
print(model.score(X_test, Y_test))
print(mean_squared_error(Y_test))