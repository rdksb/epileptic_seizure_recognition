#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ### Load and prepare data for modeling purpose

def load_balance_data():
    """load data and solve class imbalance"""
    data = pd.read_csv('Epileptic Seizure Recognition.csv')
    data = data.drop('Unnamed: 0', axis=1)

    # change the labels to binary labels (seizure y=1, non-seizure y=0)
    data['y'] = data['y'].replace([2,3,4,5], 0)

    # Solve class imbalance using random oversampling strategy
    oversampling = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
    X, y = oversampling.fit_resample(data.drop('y', axis=1), data['y'])

    return X, y


def input_data(val=True):
    """load, split and scale data"""
    X, y = load_balance_data()

    # split the dataset into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.1,
                                                        shuffle=True)
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=.1,
                                                          shuffle=True)

    # centering and scaling features
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    if val:
        X_val = scale.transform(X_val)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test


def convert_to_dataset(X, y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds




