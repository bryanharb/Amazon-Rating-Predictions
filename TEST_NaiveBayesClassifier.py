# -*- coding: utf-8 -*-
'''
Test file for Naive Bayes classifier
@author: tobiasbraun
'''

import CLASS_NaiveBayesClassifier as NB
import numpy as np
import numpy.random as npr
import scipy.sparse as spa
import sklearn.naive_bayes as sknb

# ============================== Load data ====================================

npr.seed(seed=1)

rand_features_all = spa.random(m=1000, n=10000, density=0.001, format='csr')
rand_features_all.data = (np.round(rand_features_all.data)).astype(int)
rand_labels_all = (np.round_(((npr.rand(1000))*5))).astype(int)
rand_labels_train = rand_labels_all[0:800]
rand_labels_test = rand_labels_all[801:]
rand_features_train = rand_features_all[0:800]
rand_features_test = rand_features_all[801:]

# =================================== Fit =====================================

# Custom classifier
naiveTester = NB.NaiveBayesClassifier()
naiveTester.fit(rand_features_train, rand_labels_train)

# Sklearn classifier
naiveTester_2 = sknb.MultinomialNB()
naiveTester_2.fit(rand_features_train, rand_labels_train)

# ================================ Predict ====================================

# Custome classifier
prediction = naiveTester.predict(rand_features_test)
prediction_2 = naiveTester_2.predict(rand_features_test)

# Sklearn classifier
diff = prediction - prediction_2
percentage_diff = np.count_nonzero(diff)/diff.shape[1]

# ================================ Assertions =================================

assert percentage_diff < 0.1
