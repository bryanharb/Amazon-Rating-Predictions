# -*- coding: utf-8 -*-
'''
Main file for validation and model selection
@author: bharback
'''
# =================================== Imports =================================

from sklearn.model_selection import KFold 

# Classifiers
import CLASS_NaiveBayesClassifier as NB
from sklearn.linear_model import SGDClassifier

# Config files
import CFG_Paths as paths
import CFG_CrossValidation as cv_params

# Libraries
import LIB_CrossValidation as cv

# Standard libraries
import numpy as np
import pickle
import pandas as pd

# Currently not supported
import CLASS_LogisticRegression as LR # Stoch grad descent still not implemented
import CLASS_KNNClassifier as KNN # Currently non-feasible

# ========================== Load features and labels =========================

# Features import
set_features = pickle.load(open(paths.pickle_path_set_train,"rb"))
count_features = pickle.load(open(paths.pickle_path_count_train,"rb"))
tfidf_features = pickle.load(open(paths.pickle_path_tfidf_train,"rb"))
print("FEATURE IMPORT COMPLETED.")

# Labels import
Y = pickle.load(open(paths.pickle_score_train,"rb"))
print("LABEL IMPORT COMPLETED.")

# =============================== Create K-Folds ==============================

# Naive Bayes

kf_NB = KFold(cv_params.K_NB, shuffle = True)
folds_NB_set = list(kf_NB.split(set_features))
folds_NB_count = list(kf_NB.split(count_features))
folds_NB_tfidf = list(kf_NB.split(tfidf_features))

# Logistic Regression

kf_LR = KFold(cv_params.K_LR, shuffle = True)
folds_LR_set = list(kf_LR.split(set_features))
folds_LR_count = list(kf_LR.split(count_features))
folds_LR_tfidf = list(kf_LR.split(tfidf_features))

# SVM

kf_SVM = KFold(cv_params.K_SVM, shuffle = True)
folds_SVM_set = list(kf_SVM.split(set_features))
folds_SVM_count = list(kf_SVM.split(count_features))
folds_SVM_tfidf = list(kf_SVM.split(tfidf_features))

# =============================== Validation ==================================

# Naive Bayes

NBClassifier = NB.NaiveBayesClassifier()

print("=========================================================")
print("INITIALIZING CROSS VALIDATION FOR NAIVE BAYES CLASSIFIER.")
print("SET CROSS VALIDATION BEGINNING.")
t_set_NB, v_set_NB = cv.model_cross_validation(NBClassifier, set_features, Y, folds_NB_set)
print("SET CROSS VALIDATION COMPLETED.")
print("COUNT CROSS VALIDATION BEGINNING.")
t_count_NB, v_count_NB = cv.model_cross_validation(NBClassifier, count_features, Y, folds_NB_count)
print("COUNT CROSS VALIDATION COMPLETED.")
print("TFIDF CROSS VALIDATION BEGINNING.")
t_tfidf_NB, v_tfidf_NB = cv.model_cross_validation(NBClassifier, tfidf_features, Y, folds_NB_tfidf)
print("TFIDF CROSS VALIDATION COMPLETED.")
print("=========================================================")

# Logistic Regression

LRClassifier = SGDClassifier(loss="log", penalty="l2", max_iter=5)

print("=========================================================")
print("INITIALIZING CROSS VALIDATION FOR LOGISTIC CLASSIFIER.")
print("SET CROSS VALIDATION BEGINNING.")
t_set_LR, v_set_LR = cv.model_cross_validation(LRClassifier, set_features, Y, folds_LR_set)
print("SET CROSS VALIDATION COMPLETED.")
print("COUNT CROSS VALIDATION BEGINNING.")
t_count_LR, v_count_LR = cv.model_cross_validation(LRClassifier, count_features, Y, folds_LR_count)
print("COUNT CROSS VALIDATION COMPLETED.")
print("TFIDF CROSS VALIDATION BEGINNING.")
t_tfidf_LR, v_tfidf_LR = cv.model_cross_validation(LRClassifier, tfidf_features, Y, folds_LR_tfidf)
print("TFIDF CROSS VALIDATION COMPLETED.")
print("=========================================================")

# SVM 

SVMClassifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

print("=========================================================")
print("INITIALIZING CROSS VALIDATION FOR SVM.")
print("SET CROSS VALIDATION BEGINNING.")
t_set_SVM, v_set_SVM = cv.model_cross_validation(SVMClassifier, set_features, Y, folds_SVM_set)
print("SET CROSS VALIDATION COMPLETED.")
print("COUNT CROSS VALIDATION BEGINNING.")
t_count_SVM, v_count_SVM = cv.model_cross_validation(SVMClassifier, count_features, Y, folds_SVM_count)
print("COUNT CROSS VALIDATION COMPLETED.")
print("TFIDF CROSS VALIDATION BEGINNING.")
t_tfidf_SVM, v_tfidf_SVM = cv.model_cross_validation(SVMClassifier, tfidf_features, Y, folds_SVM_tfidf)
print("TFIDF CROSS VALIDATION COMPLETED.")
print("=========================================================")

# K Nearest Neighbors

# The process of K-Fold validation for nearest neighbors proved to be unfeasible.
# We tried to run the process overnight and it still was not done. Sklearn's
# implementation of KNN goes out of memory on validation.

# ================================= Results ===================================

print("NAIVE BAYES CLASSIFIER")
print("COUNT - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_count_NB.mean(),v_count_NB.mean()))
print("SET - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_set_NB.mean(),v_set_NB.mean()))
print("TFIDF - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_tfidf_NB.mean(),v_tfidf_NB.mean()))
print("")
print("LOGISTIC REGRESSION CLASSIFIER")
print("COUNT - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_count_LR.mean(),v_count_LR.mean()))
print("SET - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_set_LR.mean(),v_set_LR.mean()))
print("TFIDF - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_tfidf_LR.mean(),v_tfidf_LR.mean()))
print("")
print("SVM CLASSIFIER")
print("COUNT - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_count_SVM.mean(),v_count_SVM.mean()))
print("SET - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_set_SVM.mean(),v_set_SVM.mean()))
print("TFIDF - AVG TRAIN ACC: {:2.2%} AVG VAL ACC: {:2.2%}".format(t_tfidf_SVM.mean(),v_tfidf_SVM.mean()))

# ================================ Boxplots ===================================

datav_NB = np.vstack((v_set_NB, v_count_NB, v_tfidf_NB)).T
datav_LR = np.vstack((v_set_LR, v_count_LR, v_tfidf_LR)).T              
datav_SVM = np.vstack((v_set_SVM, v_count_SVM, v_tfidf_SVM)).T                      
                      
data_NB = pd.DataFrame(datav_NB, columns=["NB Set","NB Count","NB TFIDF"])
data_LR = pd.DataFrame(datav_LR, columns=["LR Set","LR Count","LR TFIDF"])
data_SVM = pd.DataFrame(datav_SVM, columns=["SVM Set", "SVM Count", "SVM TFIDF"])                               ,

# These dataframes are used to create the boxplots in the main directory.