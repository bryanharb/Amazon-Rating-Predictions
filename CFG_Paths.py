# -*- coding: utf-8 -*-
'''
Configuration file storing all relevant paths
@author: bharback
'''

# ============================== Data Paths ===================================

raw_data_path = "data/raw_data/Reviews.csv"
clean_data_csv = "data/clean_data/Reviews.csv"
split_data_path = "data/split_data"
training_data_path = split_data_path + "/train_data"
test_data_path = split_data_path + "/test_data"
training_data_csv = training_data_path + "/train.csv"
test_data_csv = test_data_path + "/test.csv"

# ============================= Pickle Paths ==================================

pickles_path = "pickles"
pickle_score_train = pickles_path + "/train_scores"
pickle_score_test = pickles_path + "/test_scores"
pickle_path_count = pickles_path + "/count_features"
pickle_path_set = pickles_path + "/set_features"
pickle_path_tfidf = pickles_path + "/tfidf_features"
pickle_path_count_train = pickle_path_count + "/train_features"
pickle_path_count_test = pickle_path_count + "/test_features"
pickle_path_set_train = pickle_path_set + "/train_features"
pickle_path_set_test = pickle_path_set + "/test_features"
pickle_path_tfidf_train = pickle_path_tfidf + "/train_features"
pickle_path_tfidf_test = pickle_path_tfidf + "/test_features"
pickle_path_optimal_model = pickles_path + "/optimal_classifier/classifier"