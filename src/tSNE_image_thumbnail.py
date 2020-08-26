# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from time import time
from pathlib import Path
from PIL import Image
from time import time
from keras.datasets import cifar10


def get_CIFAR10_data() -> np.ndarray:
    """
    total data is 60000 images
    """
    # Load the raw CIFAR-10 data
    _, (X_test, y_test) = cifar10.load_data()

    # normalize the xtrain and xtest data
    X_test = X_test.astype('float32')
    X_test /= 255.0

    return X_test

def tSNE_image(test_data: np.ndarray, n_iterations: int, learning_rate: float, plots_output_path: Path, n_components: int = 2):

    # look at distribution of selected rows of data
    # crosstab = pd.crosstab(index = test_data['label'], columns = 'count')
    # print(crosstab)

    # # drop label column (it isnt a feature) and cast the df to numpy ndarray for t-SNE algorithm
    # features = test_data.drop(columns = ['label'], axis = 1).to_numpy()
    features = test_data
    features = np.reshape(features, (10000, 3072))
    print(features.shape)

    perplexities = [5, 30, 50, 100]
    for perplexity in perplexities:
        print("Starting t-SNE on images now!")
        print("Parameters are perplexity = {}, learning rate = {}, number of iterations = {}".format(perplexity, learning_rate, n_iterations))
        tsne = TSNE(n_components = 2, init = 'random', random_state = 0, perplexity = perplexity, learning_rate = 200).fit_transform(features)

        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
        width = 4000
        height = 3000
        max_dim = 100
        full_image = Image.new('RGB', (width, height))
        for idx, x in enumerate(features):
            tile = Image.fromarray(np.uint8(x * 255), 'RGB')
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                                int(tile.height / rs)),
                            Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                                    int((height-max_dim) * ty[idx])))
        filename = "tsne_perplex%d_plot.png" % (perplexity)
        fullpath = plots_output_path.joinpath(filename).resolve()
        full_image.save(str(fullpath))

if __name__ == "__main__":

    # data as numpy arrays
    x_test= get_CIFAR10_data()
    print('## Numpy Array Shapes ##')
    print('Test data shape: ', x_test.shape)

    

    #the filepath where you want to save plots to on your local machine
    plots_output_path = Path('../data/processed/tSNE_plots').resolve()

    # run tSNE and save plots to output path
    tSNE_image(x_test, 1000, 200, plots_output_path, 2)