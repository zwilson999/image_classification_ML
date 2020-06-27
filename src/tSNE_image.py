# import numpy as np # linear algebra
# import os
# import matplotlib.pyplot as plt
# import argparse

# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from time import time
# from PIL import Image
# from keras.datasets import cifar10
# from keras.utils import to_categorical

 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from time import time
from pathlib import Path



def load_label_names():
    """
    Provide Label names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_CIFAR_batch(file_name: Path):
    """ load single batch of cifar """
    with open(file_name, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes')
        X = datadict[b'data'] # image data in binary
        Y = datadict[b'labels'] # class labels i.e. the target classification
        X = X.reshape(10000, 3072) # dimension of resulting numpy array in the 'data' part of the dictionary from load_pickle()
        Y = np.array(Y)

        return X, Y

def load_CIFAR10(root_path):
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

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 10000):
    """
    total data is 60000 images, 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../data/raw/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # normalize the xtrain and xtest data
    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0


    return X_train, y_train, X_test, y_test

def pandify(xtrain: np.ndarray, ytrain: np.ndarray, xtest:np.ndarray, ytest: np.ndarray) -> pd.DataFrame:
    xtrain_df = pd.DataFrame(data = xtrain[0:, 0:],
                                index = [i for i in range(xtrain.shape[0])], # rows
                                columns = ['pixel' + str(i + 1) for i in range(xtrain.shape[1])]) # columns

    ytrain_df = pd.DataFrame(data = ytrain[0:],
                                index = [i for i in range(ytrain.shape[0])], # rows
                                columns = ['label']) # columns

    xtest_df = pd.DataFrame(data = xtest[0:, 0:],
                                index = [i for i in range(xtest.shape[0])], # rows
                                columns = ['pixel' + str(i + 1) for i in range(xtest.shape[1])]) # columns

    ytest_df = pd.DataFrame(data = ytest[0:],
                                index = [i for i in range(ytest.shape[0])], # rows
                                columns = ['label']) # columns

    return xtrain_df, ytrain_df, xtest_df, ytest_df

def create_master_data(xtrain_df: pd.DataFrame, ytrain_df: pd.DataFrame, xtest_df: pd.DataFrame, ytest_df: pd.DataFrame) -> pd.DataFrame:
    # combine all data frames into master
    train_frames = [xtrain_df, ytrain_df]
    train_df = pd.concat(train_frames, axis = 1, sort = False)
    print(train_df.shape)

    test_frames = [x_test_df, y_test_df]
    test_df = pd.concat(test_frames, axis = 1, sort = False)
    print(test_df.shape)


    # master frame of all data
    all_frames = [train_df, test_df]
    master_df = pd.concat(all_frames, sort = False)
    print(master_df.shape)

    return master_df

def tSNE_image(master_data: pd.DataFrame, n_rows_selected: int, n_iterations: int, perplexities: list, learning_rate: float, 
               plots_output_path: Path, n_components: int = 2):

    df_new = master_data[:n_rows_selected]
    label = df_new.label
    #look at distribution of selected rows of data
    crosstab = pd.crosstab(index = df_new['label'], columns='count')
    print(crosstab)

    #produce plots
    for perplexity in perplexities:
        X = df_new
        t0 = time()
        tsne = TSNE(n_components = n_components, init = 'random', random_state = 0, perplexity = perplexity, learning_rate = learning_rate)
        y = tsne.fit_transform(X)
        t1 = time()
        print("Perplexity = %d completed in %.2g sec" % (perplexity, t1 - t0))

        reduced_df = np.vstack((y.T, label)).T
        reduced_df = pd.DataFrame(data = reduced_df, columns = ["Dim1", "Dim2", "label"])
        reduced_df.label = reduced_df.label.astype(str)

        g = sns.FacetGrid(reduced_df, hue='label', height = 6).map(plt.scatter, 'Dim1', 'Dim2').add_legend(title = "Perplexity = %d" %perplexity)
    
        filename = "n%d_perplex%d_plot.png" % (n_rows_selected, perplexity)
        fullpath = plots_output_path.joinpath(filename)
    
        g.savefig(fullpath)

if __name__ == "__main__":

    # data as numpy arrays
    x_train, y_train, x_test, y_test = get_CIFAR10_data()
    print('## Numpy Array Shapes ##')
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    # convert numpy arrays to pandas dataframes
    x_train_df, y_train_df, x_test_df, y_test_df = pandify(x_train, y_train, x_test, y_test)
    print('## Pandas DataFrame Shapes ##')
    print('Train data shape: ', x_train_df.shape)
    print('Train labels shape: ', y_train_df.shape)
    print('Test data shape: ', x_test_df.shape)
    print('Test labels shape: ', y_test_df.shape)
    #print(x_train_df.head())
    #print(y_train_df.head())

    master_data_df = create_master_data(x_train_df, y_train_df, x_test_df, y_test_df)

    perplexities = [5, 30, 50, 100]
    plot_output_path = Path("../data/processed/tSNE_plots").resolve()

    # run tSNE and save plots to output path
    tSNE_image(master_data_df, 10000, 50000, perplexities, 200, 2)


    #the filepath where you want to save plots to on your local machine


