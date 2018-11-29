
# coding: utf-8

get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')


import sklearn as sk
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data_Reviews = pd.read_csv('Reviews.csv')


x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data_Reviews['Text'], data_Reviews['Score'], data_Reviews['Summary'], random_state = 0)

#Converting Reviews to Bag-of-words or discrete variables to work on
vect = CountVectorizer().fit(x_train)
vect.get_feature_names()[::2000]
Reviews_train_vectorized = vect.transform(x_train)

#Logistic Regression Train
np.mean(cross_val_score(LogisticRegression(), Reviews_train_vectorized, Rating_train, cv=3))

param_grid = {'C': np.logspace(-5, 5, 9)}

log_reg_grid = GridSearchCV(LogisticRegression(), param_grid, cv=3)
log_reg_fit = log_reg_grid.fit(Reviews_train_vectorized, Rating_train)

log_reg_fit.best_score_

x_test = pd.DataFrame({'text': x_test})

index = x_test['text'].index[x_test['text'].apply(np.isnan)]
x_test = x_test.dropna(axis=0, how = 'all')
print(x_test.shape)

x_test['label'] = y_test

print(x_test.shape)
x_test = x_test.dropna(axis=0, how = 'all')
print(x_test.shape)

x = x_test['text']
y = x_test['label']

vect.get_feature_names()[::2000]
x_test_vectorized = vect.transform(x)

pred = log_reg_fit.score(x_test_vectorized, y)
print(pred)



