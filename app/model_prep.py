"""
model_prep.py
Code related to forecast models
"""


# *** IMPORTS ****

from models import engine

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


# ***  ***

def build_preprocessor():
    """

    :return:
    """
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

    return preprocessor


def import_and_clean_data():
    """

    :return:
    """
    # Import test data
    df = pd.read_sql('SELECT * FROM public."Model";', con=engine)

    # Combine text features into 1 column for model pipeline compatibility
    df['name_and_blurb_text'] = df['name'] + ' ' + df['blurb']

    # Drop original 2 text feature columns
    df = df.drop(columns=['name', 'blurb'], axis=1)

    # Rearrange columns
    # cols = list(df.columns.values)
    df = df[['name_and_blurb_text',
             'goal',
             'campaign_duration',
             'latitude',
             'longitude',
             'category',
             'subcategory',
             'outcome',
             'days_to_success'
             ]]

    # Add new record submitted by user
    df = df.append({'name_and_blurb_text':
                      'My Fun kickstarter! - Please give me money for stuff!',
                  'goal': 30000.0,
                  'campaign_duration': 60.0,
                  'latitude': 39.9525,
                  'longitude': -75.165,
                  'category': 'fashion',
                  'subcategory': 'jewelry'},
                 ignore_index=True
                 )

    X = df.drop(columns=['outcome', 'days_to_success'])

    preprocessor = build_preprocessor()

    # # Load pickled preprocessor
    # preprocessor_path = r'data/pickle_preprocessor.pkl'
    # with open(preprocessor_path, 'rb') as file:
    #     preprocessor = pickle.load(file)

    return preprocessor.fit_transform(X), df


def process_record():
    """

    :return:
    """
    # Load model from pickle file
    path = r'data/pickle_model.pkl'
    with open(path, 'rb') as file:
        model_knn = pickle.load(file)

    # Populate mock data
    X_transformed, df = import_and_clean_data()

    # Test on last record (recently appended)
    test_num = X_transformed.shape[0] - 1

    results = model_knn.kneighbors(X_transformed[test_num][:], n_neighbors=3,
                                   return_distance=False)

    # print(results)

    prediction = model_knn.predict(X_transformed[test_num][:])
    # print(prediction)

    prob = model_knn.predict_proba(X_transformed[test_num][:])
    # print('Probability: ', prob)
    #
    # print(df.loc[test_num])
    # print(X_transformed[test_num][:])
    # print('Shape of input: ', X_transformed[test_num][0].shape)
    return str(prediction)


# # Load model from pickle file
# path = r'data/pickle_model.pkl'
# with open(path, 'rb') as file:
#     model_knn = pickle.load(file)
#
# # Populate mock data
# X_transformed, df = import_and_clean_data()
# # print(X_transformed.shape)


# results = process_record()
# results = model_knn.kneighbors(X_transformed[test_num][:], n_neighbors=3,
#                                return_distance=False)

# print(results)

# prediction = model_knn.predict(X_transformed[test_num][:])
# print(prediction)
#
# prob = model_knn.predict_proba(X_transformed[test_num][:])
# print('Probability: ', prob)
#
# print(df.loc[test_num])
# print(X_transformed[test_num][:])
# print('Shape of input: ', X_transformed[test_num][0].shape)

if __name__ == '__main__':
    print(process_record())

# Example
# pipe_knn_class.named_steps.model.kneighbors(X_test_xform[0][:],n_neighbors=3, return_distance=False)