import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.utils import to_categorical
from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from datetime import datetime
from time import time
from typing import Tuple

# set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
For model training, data needs to be in width x height x num_channels np.array format (3D image)
width = 32
height = 32
num_channels(depth) = 3
"""

def load_CIFAR_batch(file_name: Path) -> Tuple[np.ndarray, np.ndarray]:
    """ load single batch of cifar """
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes') # returns a dictionary with a 10000 x 3072 numpy array of uint8
        X = datadict[b'data'] # extract data
        Y = datadict[b'labels'] # extract class labels
        X = X.reshape((len(X), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (num_samples, width, height, num_channels) e.g. (50000 x 32 x 32 x 3)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(root_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    load all of cifar and process the data batches into the training set, and the test_batch into the test set.
    later will sample these to provide a dev set as well.
    """
    x_sets = []
    y_sets = []
    for i in range(1,6): # for the batches 1 -> 5
        f = root_path.joinpath('data_batch_%d' % i)
        X, Y = load_CIFAR_batch(f)
        x_sets.append(X)
        y_sets.append(Y)
    X_train = np.concatenate(x_sets) # concatenate row-wise
    Y_train = np.concatenate(y_sets) # concatenate row-wise
    del X, Y
    X_test, Y_test = load_CIFAR_batch(root_path.joinpath('test_batch')) # load test data from the test_batch file
    return X_train, Y_train, X_test, Y_test

def get_cifar_10_data(validation_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    total data is 60000 images, so using 49000 for train, 1000 for dev and 10000 for test
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # create validation data from the test data sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size) # create a validation split for tuning parameters
    # one hot encodings for target class vectors
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val

def pre_process_data() -> Tuple[np.ndarray, np.ndarray]:

    # unpickle all of the CIFAR data and convert to numpy arrays
    X_train, y_train, X_test, y_test, X_val, y_val = get_cifar_10_data(validation_size = 0.20)
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    # normalize pixels to between [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_test /= 255
    X_val /= 255

    return X_test, y_test

def tsne_image_from_pb(model_file_path: Path, xtest: np.ndarray, save_image_path: Path):

    model = load_model(model_file_path)
    feature_extractor = Model(inputs = model.inputs, outputs = model.get_layer('fc').output)
    features = feature_extractor.predict(xtest, batch_size = 64)
    print(features.shape)

    perplexities = [5, 30, 50, 100]
    for perplexity in perplexities:
        print("Starting t-SNE on images now!")
        print("Parameters are perplexity = {}".format(perplexity))
        tsne = TSNE(n_components = 2, init = 'random', random_state = 0, perplexity = perplexity, learning_rate = 250, n_iter = 10000).fit_transform(features)

        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
        # plot thumbnail version 
        width = 4000
        height = 3000
        max_dim = 100
        full_image = Image.new('RGB', (width, height))
        for idx, x in enumerate(xtest):
            tile = Image.fromarray(np.uint8(x * 255))
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                                int(tile.height / rs)),
                            Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                                    int((height-max_dim) * ty[idx])))
        filename = "CNN_tsne_perplex%d_plot_thumbnail.png" % (perplexity)
        fullpath = save_image_path.joinpath(filename).resolve()
        full_image.save(str(fullpath))

        # plot graphic version
        _, (_, y_test) = cifar10.load_data() # load built-in CIFAR-10 data for simplicity
        y_test = np.asarray(y_test)
        plt.figure(figsize = (16,12))

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(len(classes)):
            y_i = y_test == i
            plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label = classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()
        filename = "CNN_tsne_perplex%d_plot.png" % (perplexity)
        fullpath = save_image_path.joinpath(filename).resolve()
        plt.title('tSNE on CIFAR10 Test Data w Perplexity = %d' %(perplexity))
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.savefig(fullpath, bbox_inches='tight')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required = True, type = Path,
                        help = 'The folder for the model you want to benchmark.')
    parser.add_argument('--save_image_path', required = False, type = Path,
                        help = 'The path to save the t-SNE images')
    args = parser.parse_args()

    if args.save_image_path is None:
        args.save_image_path = Path('../data/processed/tSNE_plots')

    X_test, _ = pre_process_data()
    tsne_image_from_pb(args.model_folder, X_test, args.save_image_path)

if __name__ == '__main__':
    main()
