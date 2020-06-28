import os 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from pathlib import Path

"""
For model training, data needs to be in width x height x num_channels np.array format (3D image)
width = 32
height = 32
num_channels(depth) = 3
"""


def load_CIFAR_batch(file_name: Path) -> np.ndarray:
    """ load single batch of cifar """
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes')
        X = datadict[b'data'] # image data in binary
        Y = datadict[b'labels'] # class labels i.e. the target classification labels
        #print(X.shape)

        X = X.reshape((len(X), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (total, width, height, num_channels)
        #print(X.shape)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(root_path: Path) -> np.ndarray:

    """
    load all of cifar and process the data batches into the training set, and the test_batch into the test set.
    later will sample these to provide a dev set as well.
    """
    x_sets = []
    y_sets = []
    for i in range(1,6):

        f = root_path.joinpath('data_batch_%d' % i)
        
        X, Y = load_CIFAR_batch(f)

        x_sets.append(X)
        y_sets.append(Y)

    X_train = np.concatenate(x_sets)
    Y_train = np.concatenate(y_sets)

    del X, Y

    X_test, Y_test = load_CIFAR_batch(root_path.joinpath('test_batch'))
    return X_train, Y_train, X_test, Y_test

def get_cifar_10_data(num_training = 49000, num_dev = 1000, num_test = 10000) -> np.ndarray:
    """
    total data is 60000 images, so using 49000 for train, 1000 for dev and 10000 for test
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    ## Subsample the data ##
    # create validation set from 49000 to 50000 (1000 samples)
    mask = range(num_training, num_training + num_dev)
    print(mask)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # create train data set from 0 to 48999 (49000 samples)
    mask = range(num_training)
    print(mask)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # create test data set from 0 to 9999 (10000 samples)
    mask = range(num_test)
    print(mask)
    X_test = X_test[mask]
    y_test = y_test[mask]


    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_cifar(xtrain, xtest, xdev) -> np.ndarray:

    """
    Normalizing the images between [0,1]
    """

    X_train = xtrain.astype('float32')
    X_test = xtest.astype('float32')
    X_dev = xdev.astype('float32')

    # max pixel value is 255.0
    X_train /= 255.0
    X_test /= 255.0
    X_dev /= 255.0

    return X_train, X_test, X_dev


if __name__ == "__main__":

    # unpickle all of the CIFAR data and convert to numpy arrays
    x_train, y_train, x_dev, y_dev, x_test, y_test = get_cifar_10_data()
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_dev.shape)
    print('Validation labels shape: ', y_dev.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    # normalize CIFAR data
    X_train, X_dev, X_test = normalize_cifar(x_train, x_dev, x_test)

    """
    training below here
    """