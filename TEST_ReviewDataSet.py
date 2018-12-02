'''
Test file for ReviewDataSet
@author: bharback
'''

import CLASS_ReviewDataSet as rds
import CFG_paths as paths

# =========================== Importing =======================================

reviews = rds.ReviewDataset(paths.raw_data_path)

# Check that the dataframe has the correct column names
assert list(reviews.data.columns.values) == rds.column_names
    
# =========================== Cleaning ========================================

reviews.clean_text()

# Check that no rows are NA
assert reviews.data.isna()["Summary"].sum() == 0
assert reviews.data.isna()["Text"].sum() == 0

# =========================== Splitting =======================================

target_test_size = 0.2
reviews.split(target_test_size)

# Check that the train/test split has the correct proportions
test_size = reviews.test_set.shape[0]
dataset_size = reviews.data.shape[0]
tol = 0.001
assert abs(test_size/dataset_size - target_test_size) < tol