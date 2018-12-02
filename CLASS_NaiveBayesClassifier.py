# -*- coding: utf-8 -*-
'''
Custom implementation of Naive Bayes classifier
@author: tobiasbraun
'''

import numpy as np
import scipy.sparse as spa

# ============================== Class definition =============================

class NaiveBayesClassifier():

    def __init__(self):
        '''Set is_trained value to False to ensure that classifier is trained
        before predicting any reviews.'''
        self.is_trained = False

    def fit(self, train_data: spa.csr_matrix, scores: np.ndarray,
            alpha = 0.001):
        '''Train the classifier.'''
        self.is_trained = True
        self.train_data = train_data
        self.scores = scores
        self.fractions = self.get_fractions()
        self.rev_to_rating_sum = self.convert_data()
        self.cond_prob = self.calculate_conditional_prob(alpha)
        return None

    def get_fractions(self):
        '''Assign the fraction of all reviews belonging to each rating value.'''
        count = np.bincount(self.scores)[1:]
        self.fractions = count / np.sum(count)
        return self.fractions

    def convert_data(self):
        '''Create a list, in which each entry represents a sparse matrix
        with the sum of all the words of all the reviews that have the same
        rating.'''
        rev_to_rating_list = [np.where(self.scores == i) for i in range (1,6)]
        rev_to_rating = [self.train_data.tocsr()[rev_to_rating_list[i]]
        for i in range(5)]
        rev_to_rating_sum = [rev_to_rating[i].sum(axis = 0) for i in range(5)]
        return rev_to_rating_sum

    def calculate_conditional_prob(self,
                                   a = 0.001):
        '''Calculate the conditional probability of a word appearing in each
        rating category. Includes Laplace Smoothing with low ɑ.'''
        size = self.train_data.get_shape()[1]
        temp = self.rev_to_rating_sum
        self.cond_prob = [spa.csr_matrix((temp[i]+a)/(temp[i].sum()+1+size))
                        for i in range(5)]
        # for use in the matrix operations of fit_1
        self.log_cond_prob_trans = [(spa.csr_matrix(np.log((temp[i]+a)/
                                (temp[i].sum()+1+size)))).transpose()
                        for i in range(5)]
        return self.cond_prob

    def predict(self, x: spa.csr_matrix):
        '''Return a matrix with the predicted ratings. Applies logs to
        avoid underflow and to take into account that the probability of
        appearance of a word increases if the same word has already appeared
        before. Uses only matrix operations.'''
        if self.is_trained == False:
            return ('''The Classifier has not been trained. Please use
                    train(train_data: spa.csr_matrix, scores: np.ndarray,
                    Laplace_alpha) to train the Classifier.''')
        else:
            x.data = np.log(x.data + 1)
            self.log_cond_prob_matrix = spa.hstack(self.log_cond_prob_trans)
            log_freq = np.log(self.fractions)
            pre_final_result = x.dot(self.log_cond_prob_matrix) + log_freq
            final_prediction = (pre_final_result.argmax(axis=1) + 1).transpose()
            x.data = np.exp(x.data) - 1
            return final_prediction

    def predict_2(self, x: spa.csr_matrix):
        """Return a numpy.array with the predicted ratings. Applies logs to
        avoid underflow and to take into account that the probability of
        appearance of a word increases if the same word has already appeared before.
        Has a lower time efficiency than fit_1.
        """
        if self.is_trained == False:
            return ("""The Classifier has not been trained. Please use
                    train(train_data: spa.csr_matrix, scores: np.ndarray,
                    Laplace_alpha) to train the Classifier.""")
        else:
            final_predict = np.empty(0, dtype=int)
            for row in x:
                row_indices = spa.find(row)
                self.predictions = [1]*5
                for i in range(5):
                    for j in range(0,len(row_indices[0])):
                        self.predictions[i] *= (
                        self.cond_prob[i][row_indices[0][j],
                        row_indices[1][j]]) ** np.log(1 + row_indices[2][j])
                    self.predictions[i] *= self.fractions[i]
                smoothed_predictions = [np.log(self.predictions[i])
                                    if (self.predictions[i] != 0)
                                    else float("-inf") for i in range(5)]
                final_predict = np.append(final_predict,
                smoothed_predictions.index(max(smoothed_predictions)) + 1)
            return final_predict
