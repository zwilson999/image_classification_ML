# work in progress

import os 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import Model

from PIL import Image
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold




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

        X = X.reshape((len(X), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (len, width, height, num_channels) e.g. (50000 x 32 x 32 x 3)
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


    # append test data for full data sets
    X_test, y_test = load_CIFAR_batch(root_path.joinpath('test_batch')) # load test data from the test_batch file
    x_sets.append(X_test)
    y_sets.append(y_test)

    X = np.concatenate(x_sets) # concatenate row-wise
    Y = np.concatenate(y_sets) # concatenate row-wise


    return X, Y

def get_cifar_10_data() -> np.ndarray:
    """
    total data is 60000 images, so using 49000 for train, 1000 for dev and 10000 for test
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()

    X, Y = load_CIFAR10(cifar10_dir)

    return X, Y



def normalize_cifar(X) -> np.ndarray:

    """
    Normalizing the images between [0,1]
    """

    X = X.astype('float32')
    X  /= 255.0     # max pixel value is 255.0


    return X


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label =" test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label =" train error")
    axs[1].plot(history.history["val_loss"], label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc = "upper right")
    axs[1].set_title("Error eval")

    output_path = Path("../data/processed/evaluation_plots")
    plt.savefig(output_path.joinpath('validation_plot.png'))
    plt.show()
    plt.close()
    
def build_cnn_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape = (32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # output layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



if __name__ == "__main__":

    # unpickle all of the CIFAR data and convert to numpy arrays
    X, y = get_cifar_10_data()

    print('## Numpy Array Shapes ##')
    print('Data shape: ', X.shape)
    print('Labels shape: ', y.shape)

    # normalize CIFAR data
    X = normalize_cifar(X)


    # set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """
    training below here
    """

    # build the model with proper architecture
    CNN_model = build_cnn_model()

    # replicate the process by setting seed
    seed = 5
    np.random.seed(seed)

    # set hyperparams and fit the model
    EPOCHS = 1
    BATCH_SIZE = 64
    k = 2 # number of cross validation folds
    SAVE_MODEL_PATH = Path("../data/processed/saved_kfold_models").resolve()

    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = seed) # k fold cross validation

    cross_validation_accuracy = []
    cross_validation_error = []
    # train the model using k-fold cross validation
    for train, test in kfold.split(X, y):
        y = to_categorical(y)
        CNN_model.fit(X[train], y[train], epochs = EPOCHS, batch_size = BATCH_SIZE)

        # evaluate model on test set
        test_error, test_accuracy = CNN_model.evaluate(X[test], y[test])
        cross_validation_accuracy.append(test_accuracy)
        cross_validation_error.append(test_error)

        print("Test error: {}, test accuracy: {}".format(test_error, test_accuracy))
    # print average and standard deviation of the cross validation results
    print("Accuracy: %.2f%% (+/- %.2f%%)") % (np.mean(cross_validation_accuracy), np.std(cross_validation_accuracy))
    print("Error: %.2f%% (+/- %.2f%%)") % (np.mean(cross_validation_error), np.std(cross_validation_error))
    CNN_model.save(SAVE_MODEL_PATH)
    CNN_model.summary()
    

    ###################################
    # validate sample has proper labels
    # print(X_test[5067])
    # print(y_test[5067])
    # plt.imshow(X_test[5067])
    # plt.show()
