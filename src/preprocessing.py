import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

def label_encode(data, categories=None):
    '''
    Encode given categories(columns) using label encoding.

    Parameters:
        data: Data of type numpy.ndarray
        categories: A list(or tuple) of the columns/categories to label encode

    Returns:
        edited_data: A numpy matrix similar to data with label encoding on specified columns
    '''
    edited_data = None
    categories = set(categories)
    le = LabelEncoder()

    #Label encode each specified column
    for column in range(data.shape[1]):
        if column in categories:

            encoded_column = le.fit_transform(data[:, column])

            if type(edited_data) == type(None):
                edited_data = np.reshape(encoded_column, (encoded_column.shape[0], -1))
            else:
                edited_data = np.concatenate((edited_data, np.reshape(encoded_column, (edited_data.shape[0], -1))), axis=1)

        else:
            if type(edited_data) == type(None):
                edited_data = [data[:, column]]
            else:
                edited_data = np.concatenate((edited_data, np.reshape(data[:, column], (edited_data.shape[0], -1))), axis=1)
    return(edited_data)

def one_hot_encode(data, categories=None):
    '''
    Encode given categories(columns) using label encoding.

    Parameters:
        data: Data of type numpy.ndarray
        categories: A list(or tuple) of the columns/categories to label encode

    Returns:
        edited_data: A numpy matrix containing data with specified columns one hot encoded
    '''
    edited_data = None
    categories = set(categories)
    ohe = OneHotEncoder()
    for column in range(data.shape[1]):
        if column in categories:

            encoded_column = ohe.fit_transform(np.reshape(data[:, column], [data.shape[0], 1])).todense()

            if type(edited_data) == type(None):
                edited_data = np.reshape(encoded_column, (encoded_column.shape[0], -1))
            else:
                edited_data = np.concatenate((edited_data, np.reshape(encoded_column, (edited_data.shape[0], -1))), axis=1)
        else:

            if type(edited_data) == type(None):
                edited_data = [data[:, column]]
            else:
                edited_data = np.concatenate((edited_data, np.reshape(data[:, column], (edited_data.shape[0], -1))), axis=1)
    return(edited_data)

def mean_normalize_columns(data, columns):
    '''
    Normalize specified columns with mean normalization.

    Parameters:
        data: Data of type numpy.ndarray
        columns: A list(or tuple) of the columns/categories to label encode

    Returns:
        edited_data: A numpy matrix the same shape as data with specified columns normalized
    '''
    edited_data = np.copy(data)
    for column in columns:
        column_mean = np.mean(data[:, column])
        column_std = np.std(data[:, column])
        edited_data[:, column] = np.subtract(data[:, column], column_mean)
        edited_data[:, column] = np.divide(edited_data[:, column], column_std)


    return(edited_data)
def train_test_stratified_split(data, data_labels, test_size=0.2):
    '''
    Stratify split data into testing and training sets. Stratified sampling first
    splits up data into homogenous subgroups(strata). Then to split the data while keeping
    both the training set and testing set representative of the whole data set, samples are
    randomly taken from each strata and placed into both the training and testing sets

    Parameters:
        data: Entire input dataset of type numpy.ndarray
        data_labels: Labels corresponding to the input data(also type numpy.ndarray)
        test_size: Percentage of input data wanted in testing set.

    Returns:
        train_data, train_labels, test_data, test_labels
    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data_labels):
        train_data, train_labels = data[train_index], data_labels[train_index]
        test_data, test_labels = data[test_index], data_labels[test_index]

    return train_data, train_labels, test_data, test_labels
