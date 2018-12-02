"""
@author: yuxingchen
"""
############################ imports #########################################
import numpy as np
from scipy.sparse import rand
import CLASS_KNNClassifier as KNN
from sklearn.neighbors import KNeighborsClassifier

# training_features is of train_n * train_m
# testing_features is of test_n * train_m
train_m = 1000  
train_n = 50
test_n = 30 
k = 20

# randomly generate training and testing data
train_features = rand(train_n, train_m, density = 0.2, format = 'csr')
train_features.data[:] = 1
test_features = rand(test_n, train_m, density = 0.2, format = 'csr')
test_features.data[:] = 1
train_scores = np.random.randint(6, size = (train_n, 1))
test_scores = np.random.randint(6, size = (test_n, 1))

############################## KNNClassifier ###################################
KNNTester = KNN.KNNeighbors(k)
KNNTester.fit(train_features, train_scores)
predicted_scores1 = KNNTester.predict(test_features)
avg_performance1 = np.average(predicted_scores1 == test_scores)

##################KNeighborsClassifier from sklearn.neighbors###################
neigh = KNeighborsClassifier(n_neighbors = k)
neigh.fit(train_features, train_scores) 
predicted_scores2 = KNNTester.predict(test_features)
avg_performance2 = np.average(predicted_scores2 == test_scores)

assert avg_performance1 == avg_performance2
