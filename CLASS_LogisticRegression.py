# -*- coding: utf-8 -*-
'''
Custom implementation of Logistic Regression using stochastic gradient descent.
Currently NOT working on the review data. This class was given as part of 
the ML course IEOR4525. It has been edited and readapted for 
usability and readability.
@author: Manuel Balsera with edits from bharback
'''

import LIB_LogisticRegression as LR
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as spa

# ============================== Class definition =============================

class LogisticGDClassifier:
    
    def __init__(self,
        penalty = 0,
        learning_rate = 0.005,
        batch_size = 100,
        tol = 1e-4,
        max_iter = 100,
        verbose = True
    ):
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
    def fit(self, X: spa.csr_matrix, Y: spa.csr_matrix,
            X_val = None,
            Y_val = None):
        ''' Fit model to data. X_val and Y_val are only used to report accuracy
        during optimization they do not affect the fitted W,b parameters'''
        if X.ndim == 1:
            X = X.reshape(1,-1)
        N,D = X.shape     
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(Y)      
        K = len(self.encoder.classes_)
        Z = np.zeros((N,K),dtype=int)
        Z[np.arange(N),y] = 1
        if not(X_val is None):
            N_val = len(X_val)
            y_val = self.encoder.transform(Y_val)  
            Z_val = np.zeros((N_val,K),dtype=int)
            Z_val[np.arange(N_val),y_val] = 1
        else: 
            Z_val=None
        b_guess = np.zeros(K)
        W_guess = np.random.normal(0,1,(K,D))/np.sqrt(D)    
        self.b,self.W = LR.optimize_logistic_weights(X,Z,b_guess,W_guess,
                                                     X_val = X_val,
                                                     Z_val = Z_val,
                                                     penalty = self.penalty,
                                                     learning_rate = self.learning_rate,
                                                     batch_size = self.batch_size,
                                                     tol = self.tol,
                                                     max_iter = self.max_iter,
                                                     verbose = self.verbose
                                                     )
        # This can be substituted with LR.optimize_logistic_weights_scipy
        # to avoid using stochastic gradient descent. Unfeasible for
        # large datasets such as reviews.
        
    def predict(self, X: spa.csr_matrix):
        ''' Predict class of X'''
        if X.ndim == 1:
            X = X.reshape(1,-1)
        N,D = X.shape   
        class_probs = LR.logisticClassProbability(X,self.b,self.W)        
        y = class_probs.argmax(axis=1)
        return self.encoder.inverse_transform(y)
        

        


        
        
        
    
