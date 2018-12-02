# -*- coding: utf-8 -*-
'''
Test file for KNNClassifier
@author: yuxingchen
'''

import numpy as np
from scipy.sparse import rand
import CLASS_KNNClassifier as KNN
from sklearn.neighbors import KNeighborsClassifier

# ================================ Load data ==================================

# training_features are size of train_n * train_m
# testing_features are of size test_n * train_m
train_m = 1000  
train_n = 50
test_n = 30 
k = 20

# randomly generate training and test data
train_features = rand(train_n, train_m, density = 0.2, format = 'csr')
train_features.data[:] = 1
test_features = rand(test_n, train_m, density = 0.2, format = 'csr')
test_features.data[:] = 1
train_scores = np.random.randint(6, size = (train_n, 1))
test_scores = np.random.randint(6, size = (test_n, 1))

# =================================== Fit =====================================

# Custom classifier
KNNTester = KNN.KNNeighbors(k)
KNNTester.fit(train_features, train_scores)

# Sklearn classifier
neigh = KNeighborsClassifier(n_neighbors = k)
neigh.fit(train_features, train_scores) 

# ================================= Predict ===================================

# Custom classifier
predicted_scores1 = KNNTester.predict(test_features)
avg_performance1 = np.average(predicted_scores1 == test_scores)

# Sklearn classifier
predicted_scores2 = KNNTester.predict(test_features)
avg_performance2 = np.average(predicted_scores2 == test_scores)

# ================================ Assertions =================================

assert avg_performance1 == avg_performance2