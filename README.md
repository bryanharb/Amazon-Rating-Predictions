# Amazon-Rating-Predictions
Machine learning models for text based labeling
Group name: __
Section: 1

########## Objective

Training a model to predict the number of stars (from 1 to 5) on an Amazon review based on the text content of the review.

########## Data

The data that we used to validate our models and train the optimal model is available at
https://www.kaggle.com/snap/amazon-fine-food-reviews
The Reviews.csv file should be placed in data/raw_data.

########## File structure

We used an OOP approach to the problem, following strict conventions to ensure encapsulation. There are 5 different types of code files:
- CFG files: these files contain hardcoded values that are useful for other applications, such as paths or parameters used in other files.
- CLASS files: these files define custom classes.
- LIB files: these files contain definitions of auxiliary functions used in other scripts.
- MAIN files: the only files that should be run. These files complete all the operations needed to clean the data, create text
              representations, validate the models, train, etc.
- TEST files: these files test the class implementations with the same name.

########## Classification models

We focused our analysis on four different models: K-Neighbors, Naive Bayes, Logistic Regression and Support Vector Machines.
We implemented KNN and Naive Bayes autonomously (CLASS_NaiveBayesClassifier, CLASS_KNNClassifier) and tried to implement Logistic 
Regression (CLASS_LogisticRegression), but failed to do so due to problems with the implementation of Stochastic Gradient Descent (SGD).
We then fell back to sklearn's implementation of SGDClassifier for Logistic Regression and SVM using SGD.

########## Text representations

We focused on Bag of Words (BoW) models for text representations, considering count, set and TFIDF representations. Due to the size of
the reviews, we decided to focus only on the content of the summaries. Using summaries only, the size of the vocabulary is around
30.000 words, which is still treatable using sparse matrices. We considered stemming and removing stop-words, but this was time consuming
and did not consistently improve performance. For this reason, we decided to just work with the raw text representations of the summaries.
These word representations are created through the CLASS_ReviewDataSet class and are stored in pickles available in the pickles directory.
The whole data processing and creation of representation is handled by running MAIN_DataPipeline.

########## Validation results

We performed 10-fold cross validation on 80% of the dataset for Naive Bayes, Logistic Regression and Support Vector Machines. We did NOT
perform cross validation on K-Neighbors - this process proved to be extremely time consuming (as expected from the size of the data). Due 
to low performance on the dataset (imbalanced classes, size, etc. all work against this classifier) we decided not to bring this forward.
The result of the cross validation in terms of accuracy was the following (this may slightly vary rerunning MAIN_CrossValidation):

NAIVE BAYES CLASSIFIER

COUNT - AVG TRAIN ACC: 73.94% AVG VAL ACC: 71.44%

SET - AVG TRAIN ACC: 73.91% AVG VAL ACC: 71.42%

TFIDF - AVG TRAIN ACC: 71.17% AVG VAL ACC: 69.20%


LOGISTIC REGRESSION CLASSIFIER

COUNT - AVG TRAIN ACC: 70.22% AVG VAL ACC: 70.03%

SET - AVG TRAIN ACC: 70.23% AVG VAL ACC: 70.03%

TFIDF - AVG TRAIN ACC: 68.68% AVG VAL ACC: 68.61%


SVM CLASSIFIER

COUNT - AVG TRAIN ACC: 70.51% AVG VAL ACC: 70.13%

SET - AVG TRAIN ACC: 70.46% AVG VAL ACC: 70.06%

TFIDF - AVG TRAIN ACC: 69.28% AVG VAL ACC: 69.09%


The results of our validation can be seen with more accuracy in the OUTPUT png files showing boxplots for the various classifiers
and representations. Following this analysis, we decided to opt for a Naive Bayes classifier using the count representation for
text features.

########## Test results

We trained the Naive Bayes classifier on the entirety of the training set using count features and obtained an accuracy on the test
set of 71.68%, which is in line with the average validation accuracy. We consider this result to be satisfactory, particularly
considering the number of classes (5) and the unbalance in the dataset (5-star reviews are much more frequent than any other). This
result can be reproduced by running the MAIN_OptimalModelTest file, which also stores the optimal model in a pickle in the 
optimal_classifier directory within the pickles folder.

########## Run instructions

To run the code in its entirety, you need to run, in the following order:

MAIN_Datapipeline: cleans and splits the data, creates pickles for set, count, tfidf representations that are used during the validation phase.
MAIN_CrossValidation: performs cross validation on the models listed above and prints their performances.
MAIN_OptimalModelTest: trains the Naive Bayes classifier on the entire training set, evaluates its accuracy on the test set and pickles the optimal classifier for reuse.

