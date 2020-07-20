import os 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.manifold import TSNE

from pathlib import Path
from sklearn.model_selection import train_test_split

# global set for saving the model path
SAVE_MODEL_PATH = Path("../data/processed/saved_models").resolve()


"""
For model training, data needs to be in width x height x num_channels np.array format (3D image)
width = 32
height = 32
num_channels(depth) = 3
"""

"""This script is just testing that I get similar results by loading the built in 
keras.datasets.cifar10 dataset compared to my original script with my own data processing steps"""


# def load_CIFAR_batch(file_name: Path) -> np.ndarray:
#     """ load single batch of cifar """
#     with open(file_name, 'rb') as f:
#         datadict = pickle.load(f, encoding = 'bytes') # returns a dictionary with a 10000 x 3072 numpy array of uint8
#         X = datadict[b'data'] # extract data
#         Y = datadict[b'labels'] # extract class labels
#         #print(X.shape)

#         X = X.reshape((len(X), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (num_samples, width, height, num_channels) e.g. (50000 x 32 x 32 x 3)
#         #print(X.shape)
#         Y = np.array(Y)
#         return X, Y

# def load_CIFAR10(root_path: Path) -> np.ndarray:

#     """
#     load all of cifar and process the data batches into the training set, and the test_batch into the test set.
#     later will sample these to provide a dev set as well.
#     """
#     x_sets = []
#     y_sets = []
#     for i in range(1,6): # for the batches 1 -> 5

#         f = root_path.joinpath('data_batch_%d' % i)
#         X, Y = load_CIFAR_batch(f)

#         x_sets.append(X)
#         y_sets.append(Y)

#     X_train = np.concatenate(x_sets) # concatenate row-wise
#     Y_train = np.concatenate(y_sets) # concatenate row-wise

#     del X, Y

#     X_test, Y_test = load_CIFAR_batch(root_path.joinpath('test_batch')) # load test data from the test_batch file
#     return X_train, Y_train, X_test, Y_test


def normalize_cifar(xtrain, xtest, xval) -> np.ndarray:

    """
    Normalizing the images between [0,1]
    """

    X_train = xtrain.astype('float32')
    X_test = xtest.astype('float32')
    X_val = xval.astype('float32')

    # standardize data with z-score

    train_mean = np.mean(X_train, axis = (0,1,2,3))
    test_mean = np.mean(X_test, axis = (0,1,2,3))
    val_mean = np.mean(X_val, axis = (0,1,2,3))
    train_std_dev = np.std(X_train, axis = (0,1,2,3))
    test_std_dev = np.std(X_test, axis = (0,1,2,3))
    val_std_dev = np.std(X_val, axis = (0,1,2,3))

    X_train = (X_train - train_mean) / (train_std_dev + 1e-7)
    X_test = (X_test - test_mean) / (test_std_dev + 1e-7)
    X_val = (X_val - val_mean) / (val_std_dev + 1e-7)


    return X_train, X_test, X_val

def load_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

# def get_cifar_10_data(validation_size) -> np.ndarray:
#     """
#     total data is 60000 images, so using 49000 for train, 1000 for dev and 10000 for test
#     """
#     # Load the raw CIFAR-10 data
#     cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()

#     X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

#     # create validation data from the test data sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size) # create a validation split for tuning parameters

#     # one hot encodings for target class vectors
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)
#     y_val = to_categorical(y_val)


#     return X_train, y_train, X_test, y_test, X_val, y_val

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace = 0.5) # space the subplots apart

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc = "lower right")
    axs[0].set_title("Accuracy eval")
    axs[0].grid()

    # create error sublpot
    axs[1].plot(history.history["loss"], label =" train error")
    axs[1].plot(history.history["val_loss"], label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc = "upper right")
    axs[1].set_title("Error eval")
    axs[1].grid()

    output_path = Path("../data/processed/evaluation_plots")
    plt.savefig(output_path.joinpath('validation_plot.png'))
    plt.show()
    plt.close()
    
def pre_process_data():

    # unpickle all of the CIFAR data and convert to numpy arrays

    X_train, y_train, X_test, y_test = load_dataset()
    # create validation data from the test data sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10) # create a validation split for tuning parameters
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    X_train, X_test, X_val = normalize_cifar(X_train, X_test, X_val)

    return X_train, y_train, X_test, y_test, X_val, y_val



def learning_rate_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

def build_cnn_model():
    """
    This is the CNN model's architecture
    """
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding = 'same', input_shape = (32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # model.add(Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(weight_decay), padding='same'))
    # model.add(Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(weight_decay), padding='same'))
    # model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    #model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
    # output layer
    model.add(Dense(10, activation = 'softmax'))

    # optimize and compile model
    opt = Adam(learning_rate = 1e-3)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

def train_model(xtrain, ytrain, xtest, ytest, xval, yval, batch_size: int, num_epochs: int):
    """
    train the model and save it to a model path that is set globally
    """
    # build the model with architecture
    CNN_model = build_cnn_model()

    data_generator = ImageDataGenerator(
        rotation_range = 15,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True
        #zoom_range = [0.5, 1.0] # half zoom to double zoom possibilities
    )
    data_generator.fit(xtrain)
    
    # train the model with validation data to fine tune parameters
    # history = CNN_model.fit(xtrain, ytrain, epochs = num_epochs, batch_size = batch_size, validation_data = (xval, yval), callbacks = [LearningRateScheduler(learning_rate_schedule)])
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 25)
    history = CNN_model.fit(data_generator.flow(xtrain, ytrain, batch_size = batch_size), \
                                                                          epochs = num_epochs, \
                                                                          validation_data = (xval, yval), \
                                                                          callbacks = [LearningRateScheduler(learning_rate_schedule), es])
    # evaluate model on test set
    test_error, test_accuracy = CNN_model.evaluate(xtest, ytest)
    print("Test error: {}, test accuracy: {}".format(test_error, test_accuracy))

    CNN_model.save(SAVE_MODEL_PATH)
    CNN_model.summary()

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required = True, type = int,
                        help = 'The batch size for the CNN.')
    parser.add_argument('--epochs', required = True, type = int,
                        help = 'The number of epochs you would like to train for.')
    args = parser.parse_args()

    if args.batch_size is None:
        exit("Please provide batch size for CNN.")
        
    if args.epochs is None:
        exit("Please provide number of epochs to run model for.")

    # set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    X_train, y_train, X_test, y_test, X_val, y_val = pre_process_data()
    model_history = train_model(X_train, y_train, X_test, y_test, X_val, y_val,  args.batch_size, args.epochs)

    # plot model error and accuracy over period of epochs
    plot_history(model_history)

if __name__ == "__main__":
    main()


