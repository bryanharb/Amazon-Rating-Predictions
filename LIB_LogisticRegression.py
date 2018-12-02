# -*- coding: utf-8 -*-
'''
Useful functions for Logistic regression. These functions were given as part of 
the ML course IEOR4525. They have been slightly edited and readapted for 
usability and readability.
@author: Manuel Balsera with edits from bharback
'''

import numpy as np
import random
from scipy import optimize
import scipy.sparse as spa

def logisticClassProbability(X: spa.csr_matrix, b: np.array, W: np.array):
    '''Returns Class probabilities given data and parameters'''
    logits = X.dot(W.T)+b
    elogits = np.exp(logits-logits.max(axis=1)[:,np.newaxis]) # make calculation more stable numerically
    elogits_sum = elogits.sum(axis=1)
    class_probs = elogits/elogits_sum[:,np.newaxis]
    return class_probs

def logisticLoss(X: spa.csr_matrix, Z: spa.csr_matrix, b: np.array, W: np.array):
    '''Returns loss given data, labels and parameters'''
    class_probs = logisticClassProbability(X,b,W)
    loss = np.sum(-(Z*np.log(np.maximum(class_probs,1e-10))).sum(axis=1))
    return loss

def logisticGradient(X: spa.csr_matrix, Z: spa.csr_matrix, b: np.array, W: np.array):
    '''Returns gradient of logistic loss'''
    class_probs = logisticClassProbability(X,b,W)
    delta = (Z-class_probs)
    return -delta.sum(axis=0),-np.dot(delta.T,X)

def val_func(x0: np.array, X: spa.csr_matrix, Z: spa.csr_matrix, penalty: float):
    '''Returns loss at a point with penalty'''
    D = X.shape[1]
    K = Z.shape[1]
    b = x0[:K]
    W = x0[K:].reshape((K,D))
    loss = logisticLoss(X,Z,b,W)
    if penalty > 0:
        loss += 0.5*penalty*(W**2).sum()
    return loss

def grad_func(x0: np.array, X: spa.csr_matrix, Z: spa.csr_matrix, penalty: float):
    '''Returns gradient of loss calculated at x0'''
    D = X.shape[1]
    K = Z.shape[1]
    b = x0[:K]
    W = x0[K:].reshape((K,D))
    gradb,gradW = logisticGradient(X,Z,b,W)
    if penalty > 0: # optimization, do not perform the sum if penalty==0 
        gradW += penalty*W
    return np.concatenate([gradb,gradW.ravel()])

def generate_logistic_multinomial(X: spa.csr_matrix, W: np.array, b: np.array):
    X1 = np.c_[np.ones(len(X)),X]
    bW = np.c_[b,W]
    nu = np.dot(X1,bW.T)
    enu = np.exp(nu)
    enu_sum = enu.sum(axis=1)
    pi = enu/enu_sum[:,np.newaxis]
    Z = np.empty_like(pi)
    for i1 in range(len(pi)):
        Z[i1] = random.multinomial(1,pi[i1],1)
    return Z

def report_function(e: float, params: np.array, X: spa.csr_matrix, 
                    Z: spa.csr_matrix, X_val: spa.csr_matrix, 
                    Z_val: spa.csr_matrix, penalty: float):
    '''Reports loss and accuracy given parameters, data and labels'''
    D = X.shape[1]
    K = Z.shape[1]
    b = params[:K]
    W = params[K:].reshape((K,D))
    loss = val_func(params,X,Z,penalty)
    class_probs = logisticClassProbability(X,b,W)
    Y_pred = class_probs.argmax(axis = 1) 
    Y = Z.argmax(axis = 1)
    train_accuracy = np.mean(Y_pred == Y)
    print("\t",e,"Loss =", loss, "Train_Accuracy", train_accuracy, end = " ")
    if not(X_val is None): # if we have a valuation set we can report
                        # how well we are doing out of sample
        val_e = val_func(params, X_val, Z_val, penalty)
        class_probs = logisticClassProbability(X_val, b, W)
        Y_pred = class_probs.argmax(axis = 1)          
        Y_val = Z_val.argmax(axis = 1)     
        val_accuracy = np.mean(Y_pred == Y_val)
        print("Evaluation Loss =", val_e, "Accuracy =", val_accuracy)
    else:
        print()

def optimize_logistic_weights(X,Z,b,W,
                            X_val=None,
                            Z_val=None,
                            penalty=0,
                            learning_rate=0.01,
                            tol=1e-8,
                            max_iter=1000,
                            batch_size=100,
                            verbose=True
                              ):
    '''Logistic Regression using Stochastic Gradient Descent.
    Currently NOT working for our implementation.'''
    D = X.shape[1]
    K = Z.shape[1]
    x = np.concatenate((b, W.ravel()))
    if Z_val is not None:
        Y_val=Z_val.argmax(axis = 1)
    N = X.shape[0]
    l0 = val_func(x, X, Z, penalty)
    for e in range(max_iter):
        if (e%(max_iter//10) == 0 and verbose):
            report_function(e, x, X, Z, X_val, Z_val, penalty)
        perm = np.random.permutation(N)
        for i in range(0,N,batch_size):
            Xb = X[perm[i:i+batch_size]]
            Zb = Z[perm[i:i+batch_size]]
            p = Xb.shape[0]/N*penalty
            grad = grad_func(x,Xb,Zb,p)
            x = x - learning_rate*grad
        l = val_func(x,X,Z,penalty)
        d = np.abs(l-l0)
        if d < tol * l0:
            break
        l0=l  
    if verbose:
        report_function(e,x,X,Z,X_val,Z_val,penalty)
    b = x[:K]
    W = x[K:].reshape((K,D))
    return b,W  

def optimize_logistic_weights_scipy(X,Z,b,W,penalty=0,
                                    method = "newton-cg",
                                    tol = 1e-16,
                                    max_iter = 100):
    '''Logistic regression using scipy.optimize. Exact, but very slow.'''
    D=X.shape[1]
    K=Z.shape[1]
    x0=np.concatenate((b,W.ravel()))
   
    fit=optimize.minimize(val_func, x0, args=(X,Z,penalty),
                          jac=grad_func,
                          method=method,   
                          tol=tol)
    x1=fit.x
    b=x1[:K]
    W=x1[K:].reshape((K,D))
    return b,W


