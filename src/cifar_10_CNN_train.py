import os 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import Model
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


def load_CIFAR_batch(file_name: Path) -> np.ndarray:
    """ load single batch of cifar """
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes') # returns a dictionary with a 10000 x 3072 numpy array of uint8
        X = datadict[b'data'] # extract data
        Y = datadict[b'labels'] # extract class labels
        #print(X.shape)

        X = X.reshape((len(X), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (num_samples, width, height, num_channels) e.g. (50000 x 32 x 32 x 3)
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

    X_train = np.concatenate(x_sets) # concatenate row-wise
    Y_train = np.concatenate(y_sets) # concatenate row-wise

    del X, Y

    X_test, Y_test = load_CIFAR_batch(root_path.joinpath('test_batch')) # load test data from the test_batch file
    return X_train, Y_train, X_test, Y_test

def get_cifar_10_data(validation_size) -> np.ndarray:
    """
    total data is 60000 images, so using 49000 for train, 1000 for dev and 10000 for test
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # create validation data from the test data sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .02) # 1000 features from the train set

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)


    return X_train, y_train, X_test, y_test, X_val, y_val



def normalize_cifar(xtrain, xtest, xval) -> np.ndarray:

    """
    Normalizing the images between [0,1]
    """

    X_train = xtrain.astype('float32')
    X_test = xtest.astype('float32')
    X_val = xval.astype('float32')

    # max pixel value is 255.0
    X_train /= 255.0
    X_test /= 255.0
    X_val /= 255.0


    return X_train, X_test, X_val


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


    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
    # output layer
    model.add(Dense(10, activation = 'softmax'))

    # compile model
    opt = SGD(lr = 0.001, momentum = 0.9) # stochastic gradient descent optimizer
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model


def train_evaluate_model(batch_size: int, num_epochs: int):
    # unpickle all of the CIFAR data and convert to numpy arrays
    X_train, y_train, X_test, y_test, X_val, y_val = get_cifar_10_data(validation_size = .02)
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    # normalize CIFAR data
    X_train, X_test, X_val = normalize_cifar(X_train, X_test, X_val)

    # set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """
    training below here
    """
    # build the model with proper architecture
    CNN_model = build_cnn_model()

    # train the model with validation data to fine tune parameters
    history = CNN_model.fit(X_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_data = (X_val, y_val))

    # evaluate model on test set
    test_error, test_accuracy = CNN_model.evaluate(X_test, y_test)
    print("Test error: {}, test accuracy: {}".format(test_error, test_accuracy))

    plot_history(history)
    CNN_model.save(SAVE_MODEL_PATH)
    CNN_model.summary()


if __name__ == "__main__":

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

    train_evaluate_model(args.batch_size, args.epochs)

    # code to implement t-SNE on features from resulting model later

    # feat_extractor = Model(inputs = CNN_model.input,
    #                    outputs = CNN_model.get_layer('dense_1').output)
    # features = feat_extractor.predict(X_test, batch_size = BATCH_SIZE)

    # tsne = TSNE().fit_transform(features)
    # tx, ty = tsne[:,0], tsne[:,1]
    # tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    # ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    # width = 4000
    # height = 3000
    # max_dim = 100
    # full_image = Image.new('RGB', (width, height))
    # for idx, x in enumerate(X_test):
    #     tile = Image.fromarray(np.uint8(x * 255))
    #     rs = max(1, tile.width / max_dim, tile.height / max_dim)
    #     tile = tile.resize((int(tile.width / rs),
    #                         int(tile.height / rs)),
    #                     Image.ANTIALIAS)
    #     full_image.paste(tile, (int((width-max_dim) * tx[idx]),
    #                             int((height-max_dim) * ty[idx])))

    # full_image.save("../data/processed/tSNE_plots/CNN_TSNE.png")
    
    

    ###################################
    #validate sample has proper labels
    # print(X_test[6700])
    # print(y_test[6700])
    # plt.imshow(X_test[6700])
    # plt.show()
