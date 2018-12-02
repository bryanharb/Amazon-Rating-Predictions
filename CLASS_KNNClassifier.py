# -*- coding: utf-8 -*-
'''
Custom implementation of K-Neighbors classifier
@author: yuxingchen
'''

import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as spa

# ============================== Class definition =============================

class KNNeighbors():
 
    def __init__(self, k: int):
        self.k = k
    
    def fit(self, train_X: spa.csr_matrix, train_Y: spa.csr_matrix):
        '''Fit model to data'''
        self.train_X = train_X
        self.train_Y = train_Y    
        return None
       
    def get_neighbors(self, test_row: spa.csr_matrix):
        '''Get the k nearest neighbors'''
        # use pairwise distances
        distances = pairwise_distances(test_row, self.train_X)
        # find k smallest distances and return its indexes in training data
        indexes = np.argpartition(distances[0,], self.k)[:self.k]
        return indexes
    
    def get_freq(self):
        '''Get the frequency of each score class (i.e: [1, 2, 3, 4, 5])'''
        scores,counts= np.unique(self.train_Y, return_counts=True)
        frequency = counts/len(self.train_Y)
        return frequency
    
    def weigh_scores(self, kn_scores: np.array):
        '''Weigh the scores of k neighbors according 
        to the frequency of score classes'''
        values,counts = np.unique(kn_scores, return_counts=True)
        freq = self.get_freq()
        weighted_counts = []
        for i in range(len(counts)):
            weighted_counts.append(counts[i]/freq[values[i] - 1])      
        return values[np.array(weighted_counts).argmax()]  
    
    def predict(self, test_X: spa.csr_matrix):
        ''' Predict class of test_X'''
        test_Y=[]
        for i in range(test_X.shape[0]):
            test_row = test_X[i]
            indexes = self.get_neighbors(test_row)
            kn_scores = self.train_Y[indexes]
            test_Y.append(self.weigh_scores(kn_scores))
        return np.array(test_Y)