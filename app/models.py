"""
models.py
database schemas for Kickstarter app
"""
# *** IMPORTS ***
from os import getenv
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

import pickle
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


# from ref import DATABASE_URL

# # Comment out below when migrating to Heroku
# try:
#     from .ref import DATABASE_URL
# except ImportError:
#     from ref import DATABASE_URL
# else:
#     pass


# Create a DB Object
DB = SQLAlchemy()

# Create SQLAlchemy engine object
# postgres_url = getenv('DATABASE_URL')
# postgres_url = postgres_url.replace('postgres', 'postgresql')
# # Logic handles both local and Heroku environments
if getenv('DATABASE_URL') is None:
    try:
        from .ref import DATABASE_URL
        postgres_url = DATABASE_URL
    except:
        from ref import DATABASE_URL
        postgres_url = DATABASE_URL
else:
    postgres_url = postgres_url.replace('postgres', 'postgresql')


# try:
#     postgres_url = getenv('DATABASE_URL')
#     # Postgres URL uses the 'postgres' prefix rather than SQLAlchemy's
#     # preferred 'posgresql'. Fix this
#     postgres_url = postgres_url.replace('postgres', 'postgresql')
# except:
#     postgres_url = DATABASE_URL


engine = DB.create_engine(sa_url=postgres_url,
                          engine_opts={})

# *** DATA IMPORT/MIGRATION FUNCTIONS ***

# COMMENT OUT FOR DEPLOYMENT TO HEROKU
def csv_to_postgres(engine,
                      file: str,
                      table_name: str):
    """
    Given a *.csv filepath, create a populated table in a database
    :param engine: SQLAlchemy connection/engine for the target database
    :param file: Full filepath of the *.csv file
    :param table_name: Name of the table to be created
    :return:
    """
    df = pd.read_csv(file,
                     index_col=False)
    # print(df.head())
    # Postgres columns are case-sensitive; make lowercase
    df.columns = df.columns.str.lower()
    df.rename(columns={'unnamed: 0': 'id'},
              inplace=True)

    df.to_sql(con=engine,
              name=table_name,
              if_exists='replace',
              index=False)

    return None



# *** Define Models/Tables ***
class Kickstarter(DB.Model):
    """
    Defines the "Kickstarters" table with SQLAlchemy
    """
    id = DB.Column(DB.BigInteger, primary_key=True)
    name = DB.Column(DB.String, nullable=False)
    blurb = DB.Column(DB.String, nullable=True)
    goal = DB.Column(DB.Float, nullable=True)
    campaign_duration = DB.Column(DB.Float, nullable=True)
    current_currency = DB.Column(DB.String, nullable=True)
    fx_rate = DB.Column(DB.Float, nullable=True)
    static_usd_rate = DB.Column(DB.Float, nullable=True)
    outcome = DB.Column(DB.Boolean, nullable=True)
    days_to_success = DB.Column(DB.Float, nullable=True)
    city = DB.Column(DB.String, nullable=True)
    state = DB.Column(DB.String, nullable=True)
    country = DB.Column(DB.String, nullable=True)
    category = DB.Column(DB.String, nullable=True)
    subcategory = DB.Column(DB.String, nullable=True)
    location = DB.Column(DB.String, nullable=True)
    latitude = DB.Column(DB.Float, nullable=True)
    longitude = DB.Column(DB.Float, nullable=True)

    def __repr__(self):
        return f'Kickstarter: name - {self.name}'


# *** ML MODEL FUNCTIONS ***
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
    df = pd.read_sql('SELECT * FROM public.model10k;', con=engine)

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

    df = df.drop(columns=['outcome', 'days_to_success'])

    preprocessor = build_preprocessor()

    # # Load pickled preprocessor
    # preprocessor_path = r'data/pickle_preprocessor.pkl'
    # with open(preprocessor_path, 'rb') as file:
    #     preprocessor = pickle.load(file)

    return preprocessor.fit_transform(df)


def process_record():
    """

    :return:
    """
    # Load model from pickle file
    path = r'app/data/pickle_model.pkl'
    with open(path, 'rb') as file:
        model_knn = pickle.load(file)

    # Populate mock data
    X_transformed = import_and_clean_data()

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



# Run code to populate table
if __name__ == '__main__':
    # table_name = 'LatLong'
    # csv_to_postgres(engine=engine,
    #                 file=r'data/Kickstarter_Merged_Data_With_Lat_Lng.csv',
    #                 table_name=table_name)

    # # Query data from newly created/updated table
    # results = engine.execute(f'SELECT * FROM {table_name} limit 5;')
    # for record in results:
    #     print(record)

    table_name = 'model10k'
    csv_to_postgres(engine=engine,
                    file=r'data/Kickstarter_Data_For_Model_10k.csv',
                    table_name=table_name)

