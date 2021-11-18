"""
model_prep.py
Code related to forecast models
"""


# *** IMPORTS ****
import pickle
import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# *** Core logic ***
# Create categorical pipeline
cat_pipe = Pipeline([
    ('encoder', OneHotEncoder())
])

# Create numerical pipeline
num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

# Create text pipeline
text_pipe = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english', max_features=1000))
])

categorical = ['category', 'subcategory']
numerical = ['goal', 'campaign_duration', 'latitude', 'longitude']
text = 'name_and_blurb_text'

preprocessor = ColumnTransformer([('text', text_pipe, text),
                                  ('cat', cat_pipe, categorical),
                                  ('num', num_pipe, numerical)
                                  ])



"""TODO:
Build a mock user input to pass to the preprocessor and then load to the model
"""

# Load model from pickle file
model_knn = pickle.load('data/pickle_model_knn.pkl')

# results = model_knn.kneighbors()

# Example
# pipe_knn_class.named_steps.model.kneighbors(X_test_xform[0][:],n_neighbors=3, return_distance=False)