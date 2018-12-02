# -*- coding: utf-8 -*-
'''
Defines a class for structured review data.
The template for data will have the structure of our dataset.
The class is initialized:
- Rating: the rating associated to the review
- Summary: the review's summary
- Text: the body of the review
Summary and text need to be either strings or NaN. NaN cleaning in these
columns is handled via the clean_text method. In our current implementation,
we use exclusively summaries. 
@author: bharback
'''

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle

# ============================== Class definition =============================

class ReviewDataset():
    
    supported_representations = ["set", "count", "tfidf"]
    column_names              = ['Score', 'Summary', 'Text']
    
    def __init__(self, raw_data_path: str):            
        self.import_path = raw_data_path
        try:     
            self.data = pd.read_csv(raw_data_path)[self.column_names]
        except:
            raise ImportError("Pandas could not read a csv in the given path.")
        if list(self.data.columns.values) != ['Score', 'Summary', 'Text']:
            raise ValueError("Imported csv has wrong columns. These should be "
                             + str(self.column_names))
        self.train_set = None
        self.test_set = None      
        self.count_features = None
        self.count_test_features = None 
        self.set_features = None
        self.set_test_features = None
        self.tfidf_features = None
        self.tfidf_set_features = None
    
    def clean_text(self, 
                   clean_data_path = None, 
                   write = False):
        '''Fills the NaN in the text columns with empty strings. If 
        write is True, it writes the cleaned dataset to the path specified in 
        the clean_data_path argument'''
        self.data["Summary"] = self.data["Summary"].fillna("")
        self.data["Text"] = self.data["Text"].fillna("")
        return None
    
    def split(self, test_size: float):
        '''Divides the dataset into training and test sample. If write is
        active, it writes the training and test data sets to the paths specified
        in the training_data_path and test_data_path arguments
        '''
        train, test = train_test_split(self.data, test_size = test_size)
        summaries_all = self.data["Summary"].values
        scores_all = self.data["Score"].values
        summaries_train, summaries_test, scores_train, scores_test = train_test_split(summaries_all, scores_all, test_size = test_size)   
        self.train = train
        self.test = test
        self.summaries_train = summaries_train
        self.summaries_test = summaries_test
        self.scores_train = scores_train
        self.scores_test = scores_test             
        return None
    
    def generate_representation(self, representation: str, ):
        '''Generate representation assigns to the relevant attributes of the class
        a vector representation of the text. The nature of this representation is
        based on the representation flag.
        '''
        if representation in self.supported_representations:
            train_summary_values = self.summaries_train
            test_summary_values = self.summaries_test
            
            if representation == "count":
                count_vectorizer = CountVectorizer(input = "content")
                self.count_features = count_vectorizer.fit_transform(train_summary_values)
                self.count_test_features = count_vectorizer.transform(test_summary_values)
                        
            if representation == "set":
                set_vectorizer = CountVectorizer(input = "content", binary = True)
                self.set_features = set_vectorizer.fit_transform(train_summary_values)
                self.set_test_features = set_vectorizer.transform(test_summary_values)                  
                        
            if representation == "tfidf":
                tfidf_vectorizer = TfidfVectorizer(input = "content")
                self.tfidf_features = tfidf_vectorizer.fit_transform(train_summary_values)
                self.tfidf_test_features = tfidf_vectorizer.transform(test_summary_values)  
                        
            return None

    def save_data(self, data_path: str):
        '''Saves a csv with self.data to the given data path.'''
        try:
            self.data.to_csv(data_path)
        except:
            raise ValueError("clean_data_path has a non admissible input.")        

    def save_train_test(self, 
                        training_data_path: str, 
                        test_data_path : str):
        '''Saves the train-test split to two separate csv files with paths 
        specified by training_data_path and test_data_path'''
        try:
            self.train.to_csv(training_data_path)
            self.test.to_csv(test_data_path)
        except:
            raise ValueError("A data path argument has a non admissible input.")

    def pickle_train_test_scores(self,
                                 train_scores_path: str,
                                 test_scores_path: str):
        '''Saves two pickles with the training scores andthe test scores in the
        specified data paths.'''
        try:
            with open(train_scores_path, 'wb') as handle:
                pickle.dump(self.scores_train, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        except:
            raise ValueError("Pickle for train scores could not be written to the given path")
        try:
            with open(test_scores_path, 'wb') as handle:
                pickle.dump(self.scores_test, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        except:
            raise ValueError("Pickle for test scores could not be written to the given path")
    
        
    def pickle_representation(self, representation : str, 
                               pickle_path_train: str,
                               pickle_path_test: str):
        '''Saves a pickle with the test and training representation of the text
        given as input to the two specified data paths'''
        if representation in self.supported_representations:
            try:
                if representation == "count":              
                    with open(pickle_path_train, 'wb') as handle:
                        pickle.dump(self.count_features, handle, protocol=pickle.HIGHEST_PROTOCOL)                    
                    with open(pickle_path_test, 'wb') as handle:
                        pickle.dump(self.count_test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)               
 
                if representation == "set":
                    with open(pickle_path_train, 'wb') as handle:
                        pickle.dump(self.set_features, handle, protocol=pickle.HIGHEST_PROTOCOL)                    
                    with open(pickle_path_test, 'wb') as handle:
                        pickle.dump(self.set_test_features, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                        
                if representation == "tfidf":                        
                    with open(pickle_path_train, 'wb') as handle:
                        pickle.dump(self.tfidf_features, handle, protocol=pickle.HIGHEST_PROTOCOL)                    
                    with open(pickle_path_test, 'wb') as handle:
                        pickle.dump(self.tfidf_test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)                                  
            except:
                raise ValueError("Could not write to the specified pickle_path.")                    
            
        




        
