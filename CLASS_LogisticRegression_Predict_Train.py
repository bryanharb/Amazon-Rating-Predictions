# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:18:40 2018

@author: Arunima
""" 

from __future__ import print_function
import numpy as np
import scipy.sparse as spa
#This program uses the mxnet Module in Python that needs installation: https://mxnet.incubator.apache.org/install/
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

class LogisticRegression():
    #Class Variables:
        
    #context setting for data, or, variable placeholders where modelling is performed
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()
    
    #Variables to hold the number of inputs and outputs to minimize data overfitting and dependency
    num_inputs = 500
    num_outputs = 5
    num_examples = 60000
    
    #Allocating Random Parameters:
    W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
    b = nd.random_normal(shape=num_outputs,ctx=model_ctx)

    params = [W, b]
    
    #Attaching 'gradients' to each of the parameters eumerated:
    for param in params:
        param.attach_grad()
        
    def __init__(self, rev: spa.csr_matrix, 
                 scores: np.ndarray, x: spa.csr_matrix):
        self.rev = rev
        self.scores = scores
        self.to_predict = x
        x_train = 
        x_test = 
        
    #Data-Iterator
    batch_size = 80
    train_data = data.DataLoader(x_train, batch_size, shuffle=True)
    test_data = data.DataLoader(x_test, batch_size, shuffle=False) 
        
    def convert_data(self):
        """creates a list in which each entry represents a sparse matrices 
        with the sum of all the words of all the reviews that have the same 
        rating.""" 
        rev_to_rating_list = [np.where(self.scores == i) for i in range (1,6)]
        rev_to_rating = [self.rev.tocsr()[rev_to_rating_list[i]]
        for i in range(5)]
        rev_to_rating_sum = [rev_to_rating[i].sum(axis = 0) for i in range(5)]
        
        return rev_to_rating_sum
    
    def transform(data, label):
        """castes data and label to floats and normalizes data to range [0, 1]"""
        return data.astype(np.float32)/255, label.astype(np.float32)
    
    def sigmoid(score):
    """Inverse of Logit Function is Link Function for Linear Regression. Inverse of logit function is Sigmoid function. 
    Defining Sigmoid function on the data to convert continupus data into discrete data, for logistic Regression Operation"""
        return 1.0 / (1.0 + np.exp(-score))
    

    def softmax(y_linear):
        """linearly map our input X onto 10 different real valued outputs y_linear, 
        and normalizing to non-negative and summing to 1, before outputting"""
        exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
        norms = nd.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    sample_y_linear = nd.random_normal(shape=(2,10))
    sample_yhat = softmax(sample_y_linear)
    print(sample_yhat)
    
    #Test if all rows sum to 1: print(nd.sum(sample_yhat, axis=1))
    
    def net(X):
        """Model Definition."""
        y_linear = nd.dot(X, W) + b
        yhat = softmax(y_linear)
        return yhat
    
    def log_likelihood(self, data, features, target, weights):
    """log-likelihood: sum over all the training data for Logistic Regression."""
    #mxnet has nd.log as an inbuilt function that can be used but this method is to explain the log-likelihood algorithm
        scores = np.dot(features, weights)
        log_likelihood = np.sum( target*scores - np.log(1 + np.exp(scores)) )
        return log_likelihood
    
    def cross_entropy(yhat, y):
        """Cross-entropy maximizes the log-likelihood given to the correct labels.
           It is a Loss function, while prediction is a probability distribution. """
        #Target Y that has been formatted as a one-hot vector
        #(one value corresponding to the correct label is set to 1 and the others are set to 0)
        #only considering how much probability the prediction assigned to the correct label
        return - nd.sum(y * nd.log(yhat+1e-6))
    

    def SGD(params, lr):
        """Stochastic Gradient Descent (SGD) Optimizer"""
        for param in params:
            param[:] = param - lr * param.grad
            
    def ms_error(self, data, weights, bias):
        """Mean Squared Error Evaluation"""
        sum = 0.0
        for i in range(0, len(data)):  # walk thru each item
            X = data[i, 0:2]
            y = data[i, 2]
            z = 0.0
        
        for j in range(0, len(X)):
            z += X[j] * W[j]
            z += b
            
            prediction = Sigmoid(z)  # computed result
            sum += (prediction - y) * (prediction - y)
            
        return sum / len(data)  # mean squared error 

                   
    def evaluate_accuracy(data_iterator, net):
        numerator = 0.
        denominator = 0.
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(model_ctx).reshape((-1,784))
            label = label.as_in_context(model_ctx)
            label_one_hot = nd.one_hot(label, 10)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            numerator += nd.sum(predictions == label)
            denominator += data.shape[0]
        return (numerator / denominator).asscalar()
    
    #Test for accuracy evaluation: evaluate_accuracy(test_data, net)
    
    def predict(net,data):
            """Prediction Method."""
            output = net(data)
            return nd.argmax(output, axis=1)
    
    def train(self):
        epochs = 6
        learning_rate = .0045
        
        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, label) in enumerate(train_data):
                data = data.reshape((-1,784))
                label = label
                label_one_hot = nd.one_hot(label, 10)
                with autograd.record():
                    output = net(data)
                    loss = cross_entropy(output, label_one_hot)
                loss.backward()
                SGD(params, learning_rate)
                cumulative_loss += nd.sum(loss).asscalar()
        
        
            test_accuracy = evaluate_accuracy(test_data, net)
            train_accuracy = evaluate_accuracy(train_data, net)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
        
        
        
    