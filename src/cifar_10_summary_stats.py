import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

"""
Labels:
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
"""


def load_label_names():
    """
    Provide Label names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_CIFAR_batch(file_name: Path):

    """ load single batch of cifar """
    with open(file_name, 'rb') as f:

        datadict = pickle.load(f, encoding = 'bytes')

        features = datadict[b'data'] # 10000 samples with 3072 features (32x32x3 pixel values)
        labels = datadict[b'labels'] # labels corresponding to load_label_names ordering

        features = features.reshape((len(features), 3, 32, 32)).transpose(0, 2, 3, 1) # transpose along axis length (total, width, height, num_channels)
        labels = np.array(labels)
        return features, labels



def display_summary_stats(features: np.ndarray, labels: np.ndarray, batch_id: int, sample_id: int):
    """
    Display summary statistics of the batch id provided. Show image and label of a particular provided sample_id in that particular batch.
    Taken from and edited from this reference: https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb
    """

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))
    
    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts = True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))
    
    sample_image = features[sample_id]
    print(features[sample_id-1])
    sample_label = labels[sample_id]
    
    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    
    plt.imshow(sample_image)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_id', required = False, type = int,
                        help = 'The batch ID you want to visualize a sample from. Expecting numbers 1 - 6. 6 indicates test_batch file')
    parser.add_argument('-sample_id', required = False, type = int,
                        help = 'The sample ID that you want to visualize. Expecting number between 0 - 9999.')
    args = parser.parse_args()
    
    if args.batch_id is not None:
        if 1 <= args.batch_id <= 5:
            batch_file_path = Path("../data/raw/cifar-10-batches-py/data_batch_{}".format(args.batch_id)).resolve()
            features, labels = load_CIFAR_batch(batch_file_path)
        elif args.batch_id == 6:
            batch_file_path = Path("../data/raw/cifar-10-batches-py/test_batch").resolve()
            features, labels = load_CIFAR_batch(batch_file_path)
        else:
            exit("Expecting batch_id number between 1 and 6 (inclusive). The number you supplied was {}.".format(args.batch_id))
    else:
        pass

    if args.sample_id is not None and args.batch_id is not None:
        if 0 <= args.sample_id < len(features):
            sample_id = args.sample_id
            display_summary_stats(features, labels, args.batch_id, sample_id)
        else:
            exit("Expecting sample_id number between 1 and 10000 (inclusive). The number you supplied was {}.".format(args.sample_id))
    else:
        pass