#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Calculation of optimal accuracy for test set
@author: tobiasbraun
'''

import pickle
import CLASS_NaiveBayesClassifier as NB
import numpy as np
import CFG_Paths as paths

# ============================ Load data ======================================

count_train_features = pickle.load(open(paths.pickle_path_count_train,"rb"))
count_train_labels = pickle.load(open(paths.pickle_score_train,"rb"))
count_test_features = pickle.load(open(paths.pickle_path_count_test,"rb"))
test_labels = np.array(pickle.load(open(paths.pickle_score_test,"rb")))
print("FEATURES AND LABELS LOADED")

# =============================== Test ========================================

NBClassifier = NB.NaiveBayesClassifier()
NBClassifier.fit(count_train_features, count_train_labels)
prediction = np.array(NBClassifier.predict(count_test_features))
accuracy = np.sum(test_labels == prediction)/(len(test_labels))
print("NAIVE BAYES WITH COUNT FEATURES ACCURACY ON TEST SET: {:2.2%}".format(accuracy))

# ======================= Pickle optimal model ================================

with open(paths.pickle_path_optimal_model, 'wb') as handle:
   pickle.dump(NBClassifier, handle, protocol=pickle.HIGHEST_PROTOCOL)  
print("CLASSIFIER PICKLED SUCCESSFULLY.")   