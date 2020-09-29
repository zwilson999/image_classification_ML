import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from keras.datasets import cifar10
from keras.models import Model, load_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from datetime import datetime
from time import time
from typing import Tuple
from itertools import cycle

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

def assess_model_from_pb(model_file_path: Path, xtest: np.ndarray, ytest: np.ndarray, save_plot_path: Path):

    model = load_model(model_file_path) # load model from filepath
    feature_extractor = Model(inputs = model.inputs, outputs = model.get_layer('dense').output) # extract dense output layer (will be softmax probabilities)
    y_score = feature_extractor.predict(xtest, batch_size = 64) # one hot encoded softmax predictions
    print(y_score.shape)
    print(y_score[0:5])
    y_pred_labels = np.argmax(y_score, axis = 1) # take max softmax value to get the predicted class
    ytest_binary = label_binarize(ytest, classes = [0,1,2,3,4,5,6,7,8,9]) # one hot encode the test data true labels

    ###############FOR MANUAL CHECKING individual image results##########
    concat = np.concatenate((y_pred_labels.reshape(-1,1), ytest.reshape(-1,1)), axis = 1)
    df = pd.DataFrame(concat, columns = ['pred_labels', 'actual_labels'])
    df.to_csv(Path('../data/pred_df.csv'), index = False)
    #####################################################################

    n_classes = y_score.shape[1]
    class_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute fpr and tpr with roc_curve from the ytest true labels to the scores
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytest_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # plot each class  curve on single graph for multi-class one vs all classification
    colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.2f})'.format(lbl, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    fullpath = save_plot_path.joinpath(save_plot_path.stem +'_roc_curve.png')
    plt.savefig(fullpath)
    plt.show()

    # get confusion matrix and classification report for precision, recall, f1score at the class level
    confuse_matrix = confusion_matrix(y_true = ytest, y_pred = y_pred_labels) # confusion matrix between the ytest and y_pred indices
    class_report = classification_report(y_true = ytest, y_pred = y_pred_labels, target_names = class_labels)
    print(confuse_matrix)
    print(class_report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required = True, type = Path,
                        help = 'The folder for the model you want to benchmark.')
    parser.add_argument('--save_plot_path', required = False, type = Path,
                        help = 'Path for saving plot of ROC Curve')
    args = parser.parse_args()

    if args.save_plot_path is None:
        args.save_plot_path = Path('../data/processed/roc_curves')
    X_test, y_test = pre_process_data()

    # get classification metrics on our test data predictions
    assess_model_from_pb(args.model_folder, X_test, y_test, args.save_plot_path)

if __name__ == '__main__':
    main()
