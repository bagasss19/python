#import library
import numpy as np
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#catatan : sebenernya ada command buat deskriptif statistik cuman ga dimasukin

iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')

iris.drop('id', axis=1, inplace=True)

X = iris[['petal_len', 'petal_wd']]
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

## instantiate 
knn = KNeighborsClassifier(n_neighbors=5)
## fit 
knn.fit(X_train, y_train)


#buat kemungkinan prediksi tiap label
y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[10:12])

#buat prediksi dari data test
pred = knn.predict(X_test)
print(pred[10:12])

#untuk mengecek akurasi prediksi
print((pred==y_test.values).sum())
print(y_test.size)
#convert ke presentase
print(knn.score(X_test, y_test))

#confussion matrix 
print(confusion_matrix(y_test, pred))
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Greens)

# create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
# train model with 5-fold cv
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
# print each cv score (accuracy) 
print(cv_scores)
print(cv_scores.mean())

#setting optimal Kscore (hyperparameter)
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

#get result
knn_gscv.best_params_
knn_gscv.best_score_

#build final model
knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)
y_pred = knn_final.predict(X)
knn_final.score(X, y)

#print result
print(knn_gscv.best_score_)
print(knn_gscv.best_params_)
print(knn_final.score(X, y))

#buat prediksi dari data baru
new_data = np.array([3.76, 1.20])
new_data = new_data.reshape(1, -1)
print("ini adalah "+ str(knn_final.predict(new_data)))

new_data2 = np.array([5.76, 5.20])
new_data2 = new_data2.reshape(1, -1)
print("ini adalah "+ str(knn_final.predict(new_data2)))