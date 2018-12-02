# -*- coding: utf-8 -*-
'''
Main file for creation of data representations
@author: bharback
'''

import CLASS_ReviewDataSet as rds
import CFG_Paths as paths

# =================================== Params ==================================

target_test_size = 0.2

# ========================== Import, clean and split ==========================

print("=========================================================")
reviews = rds.ReviewDataset(paths.raw_data_path)
print("CLEAN AND SPLIT BEGINNING")
reviews.clean_text()
reviews.split(target_test_size)
print("CLEAN AND SPLIT COMPLETE")
print("=========================================================")

# ========================== Creat text representations =======================
print("=========================================================")
print("CREATION OF SET REPRESENTATION BEGINNING")
reviews.generate_representation("set")
print("CREATION OF SET REPRESENTATION COMPLETED")
print("=========================================================")
print("CREATION OF COUNT REPRESENTATION BEGINNING")
reviews.generate_representation("count")
print("CREATION OF COUNT REPRESENTATION COMPLETED")
print("=========================================================")
print("CREATION OF TFIDF REPRESENTATION BEGINNING")
reviews.generate_representation("tfidf")
print("CREATION OF TFIDF REPRESENTATION COMPLETED")
print("=========================================================")

# =============================== Save data ===================================
print("SAVING CLEAN DATA")
reviews.save_data(paths.clean_data_csv)
print("SAVING SPLIT DATA")
reviews.save_train_test(paths.training_data_csv, paths.test_data_csv)
print("PICKLING SCORES")
reviews.pickle_train_test_scores(paths.pickle_score_train, paths.pickle_score_test)
print("PICKLING COUNT REPRESENTATION")
reviews.pickle_representation("count", paths.pickle_path_count_train, 
                              paths.pickle_path_count_test)
print("PICKLING SET REPRESENTATION")
reviews.pickle_representation("set", paths.pickle_path_set_train, 
                              paths.pickle_path_set_test)
print("PICKLING TFIDF REPRESENTATION")
reviews.pickle_representation("tfidf", paths.pickle_path_tfidf_train, 
                              paths.pickle_path_tfidf_test)





