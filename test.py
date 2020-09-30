# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:36:42 2020

@author: Razi
"""


#from KNearestNeighbors import KNN
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from KNearestNeighbors import KNNClassifier

iris = datasets.load_iris()

X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(data=iris.target, columns=["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)

knn = KNNClassifier(k = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
c_report = classification_report(y_test, y_pred)
print(c_report)

print(50*"#")
# using sklearn
sknn = KNeighborsClassifier(n_neighbors=3)
sknn.fit(X_train, y_train)
sy_pred = sknn.predict(X_test)
sc_report = classification_report(y_test, sy_pred)
print(sc_report)
