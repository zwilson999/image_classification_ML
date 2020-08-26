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
from PIL import Image
from time import time

'''
This script is for visualizing subsets of the entire CIFAR10 dataset's raw features.
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

# flatten the data into data frames
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
    train_df = pd.concat(train_frames, axis = 1, sort = False) # like an outer join (left to right)
    #print(train_df.iloc[[0,25689]]) # can validate these using cifar_10_summary_stats.py, e.g. 25689, you would use summary_stats and search batch_id = 3, sample_id = 5689

    test_frames = [x_test_df, y_test_df]
    test_df = pd.concat(test_frames, axis = 1, sort = False) # like an outer join (left to right)
    #print(test_df.iloc[[0,-1]])

    labels_dict = {0: 'airplane',
                   1: 'automobile',
                   2: 'bird',
                   3: 'cat',
                   4: 'deer',
                   5: 'dog',
                   6: 'frog',
                   7: 'horse',
                   8: 'ship',
                   9: 'truck'
    }
    # master frame of all data
    all_frames = [train_df, test_df]
    master_df = pd.concat(all_frames, sort = False).reset_index(drop = True) # join all frames together row-wise and reset the index
    master_df.label = [labels_dict[item] for item in master_df.label] # convert label integers to their proper string label
    #print(master_df.tail())
    #print(master_df.shape)
    return master_df, train_df, test_df

def tSNE_image(master_data: pd.DataFrame, n_rows_selected: int, n_iterations: int, perplexities: list, learning_rate: float, 
               plots_output_path: Path, n_components: int = 2):

    X = master_data[:n_rows_selected] # select a subset of data
    label = X.label

    # look at distribution of selected rows of data
    crosstab = pd.crosstab(index = X['label'], columns = 'count')
    print(crosstab)

    # drop label column (it isnt a feature) and cast the df to numpy ndarray for t-SNE algorithm
    X = X.drop(columns = ['label'], axis = 1).to_numpy()
    # use standard scaler to scale data with mean = 0 and std deviation 1
    X = StandardScaler().fit_transform(X) 


    #produce plots
    for perplexity in perplexities:
        
        time_start = time()
        tsne = TSNE(n_components = n_components, init = 'random', random_state = 0, perplexity = perplexity, learning_rate = learning_rate)
        print("Starting t-SNE on {} images now!".format(n_rows_selected))
        print("Parameters are perplexity = {}, learning rate = {}, number of iterations = {}".format(perplexity, learning_rate, n_iterations))
        y = tsne.fit_transform(X)
        time_end = time()
        print("Perplexity = %d completed in %.2g sec" % (perplexity, time_end - time_start))

        reduced_df = np.vstack((y.T, label)).T
        reduced_df = pd.DataFrame(data = reduced_df, columns = ["Dim1", "Dim2", "label"])
        reduced_df.label = reduced_df.label.astype(str)

        g = sns.FacetGrid(reduced_df, hue='label', height = 6).map(plt.scatter, 'Dim1', 'Dim2').add_legend(title = "Perplexity = %d" %perplexity)
    
        filename = "n%d_perplex%d_plot.png" % (n_rows_selected, perplexity)
        fullpath = plots_output_path.joinpath(filename)

        g.fig.suptitle("tSNE with %d rows and perplexity = %d" % (n_rows_selected, perplexity))
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

    master_data_df, train_df, test_df = create_master_data(x_train_df, y_train_df, x_test_df, y_test_df)

    perplexities = [5, 30, 50, 100]

    #the filepath where you want to save plots to on your local machine
    plot_output_path = Path("../data/processed/tSNE_plots").resolve()

    # run tSNE and save plots to output path
    #tSNE_image(test_df, 5000, 5000, perplexities)
    tSNE_image(master_data_df, 5000, 5000, perplexities, 200, plot_output_path, 2)