# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:36:42 2020

@author: Razi
"""


#from KNearestNeighbors import KNN
from sklearn import datasets
import pandas as pd

from sklearn.model_selection import train_test_split
from KNearestNeighbors import KNNRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

boston = datasets.load_boston()

X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
y = pd.DataFrame(data=boston.target, columns=["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)

knn = KNNRegressor(k = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))

print(20*"#")

sknn = KNeighborsRegressor(n_neighbors=3)
sknn.fit(X_train, y_train)
sy_pred = sknn.predict(X_test)
print("MSE: ", mean_squared_error(y_test, sy_pred))
print("R2: ", r2_score(y_test, sy_pred))