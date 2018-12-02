# -*- coding: utf-8 -*-
'''
Library containing useful functions for cross validation
@author: bharback
'''

import numpy as np
import scipy.sparse as spa

def model_cross_validation(model, X: spa.csr_matrix , Y: spa.csr_matrix, folds,
                           verbose = False):
    '''Performs cross validation on the folds specified by folds, which should
    be defined using sklearn. Returns an array containing the accuracies computed
    on each fold. Verbose toggles a printed message after completed folds'''
    kfolds = len(folds)
    train_performance, validation_performance = np.empty(kfolds), np.empty(kfolds)
    for i in range(kfolds):
        train, validation = folds[i]
        X_train, Y_train = X[train], Y[train]
        X_validation, Y_validation = X[validation], Y[validation]
        model.fit(X_train,Y_train)
        train_accuracy = np.average(model.predict(X_train) == Y_train)
        validation_accuracy = np.average(model.predict(X_validation) == Y_validation)
        train_performance[i] = train_accuracy
        validation_performance[i] = validation_accuracy
        if verbose:
            print("Fold number {} completed.".format(i))
    return np.array(train_performance),np.array(validation_performance)