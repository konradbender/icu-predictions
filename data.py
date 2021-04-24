import pandas as pd
import random


# load training data as dataframe from csv and return all rows or random sample
def get_training_features(limit_rows=None):
    df_train_features = pd.read_csv('train_features.csv')

    # sample rows for easier debugging
    if limit_rows is not None:
        # we are sampling at random to get nicer means at the standardization
        row_indices = random.sample(range(0,df_train_features.shape[0]), limit_rows)
        df_train_features = df_train_features.iloc[row_indices, :]

    return df_train_features


# load training labels as dataframe from csv and return all
def get_training_labels():
    return pd.read_csv('train_labels.csv')

def get_test_features():
	return pd.read_csv('test_features.csv')

