#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobiasbraun
"""
############################### imports #######################################

import pickle
import random
import CLASS_NaiveBayesClassifier as naive
import numpy as np

############################# loading data ####################################

with open ("pickles/count_features/test_features","rb") as f:
    reviews_count_format=pickle.load(f)

with open ("pickles/test_scores","rb") as f:
    ratings=pickle.load(f)

########################### choosing parameters ###############################

#random.seed(1)
n = 100
m = 1000
performance_threshold = 0.73
sum_performance_1, sum_performance_2 = 0, 0

############################### testing #######################################

naiveTester = naive.NaiveBayesClassifier()
naiveTester.train(reviews_count_format, ratings)
for j in range(n):
    count_1, count_2 = 0, 0
    x = random.randint(1,100000)
    words_to_be_predicted = reviews_count_format.tocsr()[x:x+m]
    real_ratings = ratings[x:x+m]
    diff_1 = naiveTester.fit_1(words_to_be_predicted) - real_ratings
    #diff_2 = naiveTester.fit_2(words_to_be_predicted) - real_ratings
    count_1 = m - np.count_nonzero (diff_1)
    #count_2 = m - np.count_nonzero (diff_2)
    percentage_performance_1 = count_1/m
    #percentage_performance_2 = count_2/m
    sum_performance_1 += percentage_performance_1
    #sum_performance_2 += percentage_performance_2
    
    #print("Predicted (fit_1 :" + str(naiveTester.fit_1(words_to_be_predicted)))
    #print("Predicted (fit_2 :" + str(naiveTester.fit_2(words_to_be_predicted)))
    #print("Actual rating: " + str(real_ratings))
    #print(naiveTester.fit_1(words_to_be_predicted))
average_performance_1 = sum_performance_1 / n
#average_performance_2 = sum_performance_2 / n

print(average_performance_1)
assert (average_performance_1 >= performance_threshold)