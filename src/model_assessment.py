import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
#import pandas as pd
import argparse
import matplotlib.cm as cm
from keras.datasets import cifar10
from keras.models import Model, load_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from matplotlib.colors import Normalize
from datetime import datetime
from time import time
from typing import Tuple
from itertools import cycle

# set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# seeding necessary for neural network reproducibility
SEED = 123456
import os
import random as rn
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)

# set this to allow memory growth on your GPU, otherwise, CUDNN may not initialize
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
    total data is 60000 images, perform split according to provided validation_size here. Test Data will be 10000 images
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = Path('../data/raw/cifar-10-batches-py/').resolve()
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # create validation data from the test data sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size, random_state = 123456) # create a validation split for tuning parameters
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

    return X_train, y_train, X_val, y_val, X_test, y_test

def assess_roc(model, xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray, yval: np.ndarray, xtest: np.ndarray, ytest: np.ndarray, 
               save_plot_path: Path, n_classes: int, class_labels: list):
    """
    Get ROC Curve and AUC plots for Validation and Test Data
    """
    feature_extractor = Model(inputs = model.inputs, outputs = model.get_layer('dense').output) # extract dense output layer (will be softmax probabilities)

    y_train_score = feature_extractor.predict(xtrain, batch_size = 64) # softmax probabilities for training data
    y_train_pred_labels = np.argmax(y_train_score, axis = 1) # predicted labels for training data
    y_train_binary = label_binarize(ytrain, classes = [0,1,2,3,4,5,6,7,8,9]) # one-hot encoded training response data

    y_val_score = feature_extractor.predict(xval, batch_size = 64) # softmax probability for validation data
    y_val_pred_labels = np.argmax(y_val_score, axis = 1) # get validation predicted labels
    y_val_binary = label_binarize(yval, classes = [0,1,2,3,4,5,6,7,8,9]) # one-hot encoded validation response data

    y_test_score = feature_extractor.predict(xtest, batch_size = 64) # one hot encoded softmax predictions
    y_test_pred_labels = np.argmax(y_test_score, axis = 1) # take max softmax value to get the predicted class
    y_test_binary = label_binarize(ytest, classes = [0,1,2,3,4,5,6,7,8,9]) # one-hot encoded testing response data

    train_fpr = dict()
    train_tpr = dict()
    train_roc_auc = dict()
    val_fpr = dict()
    val_tpr = dict()
    val_roc_auc = dict()
    test_fpr = dict()
    test_tpr = dict()
    test_roc_auc = dict()
    
    # compute fpr and tpr with roc_curve for validation and test data to be later used in different plots
    for i in range(n_classes):
        train_fpr[i], train_tpr[i], _ = roc_curve(y_train_binary[:, i], y_train_score[:, i])
        train_roc_auc[i] = auc(train_fpr[i], train_tpr[i])
        val_fpr[i], val_tpr[i], _ = roc_curve(y_val_binary[:,i], y_val_score[:, i])
        val_roc_auc[i] = auc(val_fpr[i], val_tpr[i])
        test_fpr[i], test_tpr[i], _ = roc_curve(y_test_binary[:, i], y_test_score[:, i])
        test_roc_auc[i] = auc(test_fpr[i], test_tpr[i])

    colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(train_fpr[i], train_tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, train_roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training ROC Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('train_roc_curve.png')
    plt.savefig(fullpath)
    plt.close()

    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(val_fpr[i], val_tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, val_roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation ROC Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('val_roc_curve.png')
    plt.savefig(fullpath)
    plt.close()

    # plot each class curve on single graph for multi-class one vs all classification
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(test_fpr[i], test_tpr[i], color = color, lw = 1.5,
        label = 'ROC Curve of class {0} (area = {1:0.3f})'.format(lbl, test_roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw = 1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'lower right', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('test_roc_curve.png')
    plt.savefig(fullpath)
    plt.close()

    # get confusion matrix and classification report for precision, recall, f1score at the class level
    
    train_confuse_matrix = confusion_matrix(y_true = ytrain, y_pred = y_train_pred_labels)
    train_class_report = classification_report(y_true = ytrain, y_pred = y_train_pred_labels, target_names = class_labels)
    val_confuse_matrix = confusion_matrix(y_true = yval, y_pred = y_val_pred_labels)
    val_class_report = classification_report(y_true = yval, y_pred = y_val_pred_labels, target_names = class_labels)
    test_confuse_matrix = confusion_matrix(y_true = ytest, y_pred = y_test_pred_labels) # confusion matrix between the ytest and y_pred indices
    test_class_report = classification_report(y_true = ytest, y_pred = y_test_pred_labels, target_names = class_labels)


    # plot and save confusion matrices
    train_cm_display = ConfusionMatrixDisplay(train_confuse_matrix, display_labels = class_labels)
    train_cm_display.plot(cmap = 'Greens')
    fullpath = save_plot_path.joinpath('train_confusion_matrix.png')
    plt.title("Train Confusion Matrix")
    plt.xticks(range(len(class_labels)), class_labels, rotation = 90) # rotate x labels
    plt.yticks(range(len(class_labels)), class_labels) 
    plt.savefig(fullpath, bbox_inches = "tight")
    plt.clf()

    val_cm_display = ConfusionMatrixDisplay(val_confuse_matrix, display_labels = class_labels)
    val_cm_display.plot(cmap = "Greens") # greens color map
    fullpath = save_plot_path.joinpath('val_confusion_matrix.png')
    plt.title("Validation Confusion Matrix")
    plt.xticks(range(len(class_labels)), class_labels, rotation = 90) # rotate x labels
    plt.yticks(range(len(class_labels)), class_labels)
    plt.savefig(fullpath, bbox_inches = "tight")
    plt.clf()

    test_cm_display = ConfusionMatrixDisplay(test_confuse_matrix, display_labels = class_labels)
    test_cm_display.plot(cmap = 'Greens')
    fullpath = save_plot_path.joinpath('test_confusion_matrix.png')
    plt.title("Test Confusion Matrix")
    plt.xticks(range(len(class_labels)), class_labels, rotation = 90) # rotate x labels
    plt.yticks(range(len(class_labels)), class_labels)
    plt.savefig(fullpath, bbox_inches = "tight")
    plt.clf()

    print(train_class_report)
    print(val_class_report)
    print(test_class_report)

def plot_f1_score_curve():
    """
    Generate iso-f1 scores per sklearn documentation for showing  f1 cutoffs on PRCs
    """

    f_scores = np.linspace(0.2, 0.8, num = 4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
        plt.annotate('f1 = {0:0.1f}'.format(f_score), xy = (0.7, y[49] + 0.02))

def assess_pr_curve(model, xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray, yval: np.ndarray, xtest: np.ndarray, ytest: np.ndarray, 
                    save_plot_path: Path, n_classes: int, class_labels: list):
    """
    Get Precision and Recall and P/R Curve plots for Validation and Test data
    """

    feature_extractor = Model(inputs = model.inputs, outputs = model.get_layer('dense').output) # extract dense output layer (will be softmax probabilities)

    y_train_score = feature_extractor.predict(xtrain, batch_size = 64) # softmax probabilities for training data
    y_train_binary = label_binarize(ytrain, classes = [0,1,2,3,4,5,6,7,8,9]) # one hot encode train data

    y_val_score = feature_extractor.predict(xval, batch_size = 64) # softmax probability for validation data
    y_val_binary = label_binarize(yval, classes = [0,1,2,3,4,5,6,7,8,9]) # one hot encode validation data

    y_test_score = feature_extractor.predict(xtest, batch_size = 64) # one hot encoded softmax predictions
    y_test_binary = label_binarize(ytest, classes = [0,1,2,3,4,5,6,7,8,9]) # one hot encode the test data true labels

    # Precision-Recall Curves for train/val/test
    train_precision = dict()
    train_recall = dict()
    train_avg_precision = dict()
    train_pr_auc = dict()

    val_precision = dict()
    val_recall = dict()
    val_avg_precision = dict()
    val_pr_auc = dict()

    test_precision = dict()
    test_recall = dict()
    test_avg_precision = dict()
    test_pr_auc = dict()

    for i in range(n_classes):
        train_precision[i], train_recall[i], _ = precision_recall_curve(y_train_binary[:, i], y_train_score[:, i])
        train_avg_precision[i] = average_precision_score(y_train_binary[:, i], y_train_score[:, i])
        train_pr_auc[i] = auc(train_recall[i], train_precision[i])

        val_precision[i], val_recall[i], _ = precision_recall_curve(y_val_binary[:, i], y_val_score[:, i])
        val_avg_precision[i] = average_precision_score(y_val_binary[:, i], y_val_score[:, i])
        val_pr_auc[i] = auc(val_recall[i], val_precision[i])

        test_precision[i], test_recall[i], _ = precision_recall_curve(y_test_binary[:, i], y_test_score[:, i])
        test_avg_precision[i] = average_precision_score(y_test_binary[:, i], y_test_score[:, i])
        test_pr_auc[i] = auc(test_recall[i], test_precision[i])

    colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])

    # plot each class curve on single graph for multi-class one vs all classification
    plot_f1_score_curve()
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(train_recall[i], train_precision[i], color = color, lw = 2,
        label = 'P/R Curve of class {0} (AUC = {1:0.3f})'.format(lbl, train_pr_auc[i]))
    #plt.hlines(0, xmin = -0.02, xmax = 1.0, linestyle = 'dashed')
    plt.xlim([-0.02, 1.03])
    plt.ylim([-0.03, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Train P/R Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'center left', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('train_pr_curve_AUC.png')
    plt.savefig(fullpath)
    plt.close()


    # plot each class curve on single graph for multi-class one vs all classification
    plot_f1_score_curve()
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(val_recall[i], val_precision[i], color = color, lw = 2,
        label = 'P/R Curve of class {0} (AUC = {1:0.3f})'.format(lbl, val_pr_auc[i]))
    #plt.hlines(0, xmin = -0.02, xmax = 1.0, linestyle = 'dashed')
    plt.xlim([-0.02, 1.03])
    plt.ylim([-0.03, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Validation P/R Curve CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'center left', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('val_pr_curve_AUC.png')
    plt.savefig(fullpath)
    plt.close()

    # plot each class curve on single graph for multi-class one vs all classification
    plot_f1_score_curve()
    for i, color, lbl in zip(range(n_classes), colors, class_labels):
        plt.plot(test_recall[i], test_precision[i], color = color, lw = 2,
        label = 'P/R Curve of class {0} (AUC = {1:0.3f})'.format(lbl, test_pr_auc[i]))
    #plt.hlines(0, xmin = -0.02, xmax = 1.0, linestyle = 'dashed')
    plt.xlim([-0.02, 1.03])
    plt.ylim([-0.03, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test P/R Curve for CIFAR-10 Multi-Class Data')
    plt.legend(loc = 'center left', prop = {'size': 6})
    fullpath = save_plot_path.joinpath('test_pr_curve_AUC.png')
    plt.savefig(fullpath)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required = True, type = Path,
                        help = 'The folder for the model you want to benchmark.')
    parser.add_argument('--save_plot_path', required = False, type = Path,
                        help = 'Path for saving plot of ROC Curve')
    args = parser.parse_args()

    if args.save_plot_path is None:
        args.save_plot_path = Path('../data/processed/plots')
    X_train, y_train, X_val, y_val, X_test, y_test = pre_process_data()
    
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    n_classes = len(class_labels)
    model = load_model(args.model_folder) # load model from filepath

    # generate ROC Curve plots for validation and test data
    assess_roc(model, X_train, y_train, X_val, y_val, X_test, y_test, args.save_plot_path, n_classes, class_labels)
    # get classification metrics on our test data predictions
    assess_pr_curve(model, X_train, y_train, X_val, y_val, X_test, y_test, args.save_plot_path, n_classes, class_labels)

if __name__ == '__main__':
    main()


    #print(ytest[1])
    #print(y_test_score[1])
    #print(ytest_binary[1])
    ###############FOR MANUAL CHECKING individual image results##########
    # concat = np.concatenate((y_pred_labels.reshape(-1,1), ytest.reshape(-1,1)), axis = 1)
    # df = pd.DataFrame(concat, columns = ['pred_labels', 'actual_labels'])
    # df.to_csv(Path('../data/pred_df.csv'), index = False)

    # one hot encoded scores for all observations df
    #y_comparison_df = np.concatenate((ytest.reshape(-1,1), y_score), axis = 1)
    #print(y_comparison_df.shape)
    #y_score_df = pd.DataFrame(y_comparison_df, columns = ['actual_label','airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
    #y_score_df.to_csv(Path('../data/pred_y_score.csv'), index = False)
    #####################################################################