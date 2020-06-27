from keras.datasets import cifar10
import numpy as np

def normalize_cifar(xtrain, xtest) -> np.ndarray:

    """
    Normalizing the images between [0,1]
    """

    X_train = xtrain.astype('float32')
    X_test = xtest.astype('float32')

    # max pixel value is 255.0
    X_train /= 255.0
    X_test /= 255.0


    return X_train, X_test

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()

train_x, test_x = normalize_cifar(trainX, testX)

print(train_x[0:1])