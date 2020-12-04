import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#catatan : sebenernya ada command buat deskriptif statistik cuman ga dimasukin

#import data
boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

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

#buat prediksi
new_RM = np.array([6.5]).reshape(-1,1) # make sure it's 2d
print(model.predict(new_RM))

y_test_predicted = model.predict(X_test)
y_test_predicted.shape
type(y_test_predicted) 

#show r square
model.score(X_test, Y_test)
print(model.score(X_test, Y_test))

#visualisasi data
plt.scatter(X_test, Y_test, 
  label='testing data');
plt.plot(X_test, y_test_predicted,
  label='prediction', linewidth=3)
plt.xlabel('RM'); plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()

#regresi linear berganda


## data preparation
X2 = boston[['RM', 'LSTAT']]
Y = boston['MEDV']
## train test split
## same random_state to ensure the same splits
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y,
  test_size = 0.3,
  random_state=1)
model2 = LinearRegression()
model2.fit(X2_train, Y_train)

#hasil output
print(model2.intercept_)
print(model2.coef_)
