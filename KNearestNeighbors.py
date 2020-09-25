# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:36:42 2020

@author: Razi
"""

import numpy as np
#from statistics import multimode
class KNN:
    # constructor
    def __init__(self, k):
        #initializing the class variable k
        self.k = k

    # creating a fit method that simply stores values in class variables
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("Training Completed.")

    # method for calculating the euclidean distance between two points
    def euclidean_distance(self, point_A, point_B, length):
        distance = 0
        
        # looping through all the features for the two points
        for i in range(length):
            distance += np.square(point_A[i] - point_B[i])
        
        return np.sqrt(distance)
        
 
    # the core function of KNN for predicting the class / label / target  
    def predict(self, X_test):
        
   
        # an empty list for storing X_test final predicted values
        y_pred = []
        
        for y in X_test.index:
        
            # an empty list to contain the combination of distance and index
            matrix = []
            
            # looping through all the indices of X_train
            for x in self.X_train.index:
                
                # finding distance of points of X_test for each point in X_train
                distance = self.euclidean_distance(X_test.loc[y], self.X_train.loc[x], X_test.shape[1])
                
                # populating the list with distance and index
                matrix.append([distance, x, y])
               
            matrix = sorted(matrix)

            # keeping only the first k entries of the list
            final_matrix = matrix[:self.k]

            # creating an empty list to hold the labels
            labels = []
        
            # looping through the k indices of the list
            for i in final_matrix:
                
                # populating the list with only the class / label / target from y_train
                labels.append(self.y_train.loc[i[1]][0])
                
                results = max(set(labels), key=labels.count)
                               
                
            y_pred.append(results)
        
        return y_pred