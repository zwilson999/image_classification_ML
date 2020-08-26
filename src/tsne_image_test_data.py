import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from time import time
from pathlib import Path
from PIL import Image
from time import time
from keras.datasets import cifar10

'''
This script is for visualizing the test data of cifar10's raw features.
'''

def load_CIFAR_batch(file_name: Path) -> np.ndarray:
    """ load single batch of cifar """
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes')
        X = datadict[b'data'] # image data in binary
        Y = datadict[b'labels'] # class labels i.e. the target classification
        X = X.reshape(10000, 3072) # dimension of resulting flattened numpy array in the 'data' part of the dictionary from load_pickle()
        Y = np.array(Y)

        return X, Y

def load_CIFAR10(root_path) -> np.ndarray:
    """
    load all of cifar and process the data batches into the training set, and the test_batch into the test set.
    later will sample these to provide a dev set as well.
    """
    x_sets = []
    y_sets = []
    for i in range(1,6):
        f = os.path.join(root_path, 'data_batch_%d' % (i, ))
        X, Y = load_CIFAR_batch(f)
        x_sets.append(X)
        y_sets.append(Y)
    X_train = np.concatenate(x_sets)
    Y_train = np.concatenate(y_sets)
    del X, Y
    X_test, Y_test = load_CIFAR_batch(os.path.join(root_path, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

def get_CIFAR10_data():
    """
    total data is 60000 images
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../data/raw/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # normalize the xtrain and xtest data
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255.0
    # X_test /= 255.0

    return X_train, y_train, X_test, y_test


def tSNE_image(xtest: np.ndarray, perplexities: list, learning_rate: float, plot_output_path: Path, n_components: int = 2):

    X = xtest
    # use standard scaler to scale data with mean = 0 and std deviation 1
    X = StandardScaler().fit_transform(X) 
    _, (_, y_test) = cifar10.load_data()

    y_test = np.asarray(y_test)
    print(y_test.shape)

    # produce plots
    for perplexity in perplexities:
        time_start = time()
        print("Starting t-SNE on images now!")
        print("Parameters are perplexity = {}, learning rate = {}".format(perplexity, learning_rate))
        tsne = TSNE(n_components = n_components, init = 'random', random_state = 0, perplexity = perplexity, learning_rate = learning_rate).fit_transform(X)
        time_end = time()
        print("Perplexity = %d completed in %.2g sec" % (perplexity, time_end - time_start))

        tx, ty = tsne[:,0], tsne[:,1]
        # min max normalize the data for better plotting on 2D plot
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        plt.figure(figsize = (16,12))

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(len(classes)):
            y_i = y_test == i
            plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label = classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()
        filename = "test_data_perplex%d_plot.png" % (perplexity)
        fullpath = plot_output_path.joinpath(filename)
        plt.title('tSNE on CIFAR10 Test Data w Perplexity = %d' %(perplexity))
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.savefig(fullpath, bbox_inches='tight')


if __name__ == "__main__":

    # data as numpy arrays
    x_train, y_train, x_test, y_test = get_CIFAR10_data()
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    perplexities = [5, 30, 50, 100]
    #the filepath where you want to save plots to on your local machine
    plot_output_path = Path("../data/processed/tSNE_plots").resolve()

    # run tSNE and save plots to output path
    #tSNE_image(test_df, 5000, 5000, perplexities)
    tSNE_image(x_test, perplexities, 200, plot_output_path, 2)