# Master's Research Project
# An analysis of the CIFAR-10 Dataset using Convolutional Neural Networks and feature similarity visualization with T-distributed Stochastic Neighbor Embedding (t-SNE)

### Note this REPO utilizes GPU facilitated keras/tensorflow with CUDA/CUDNN dependencies.



## Script Utilities

```1. cifar_10_CNN_train.py```

This script is the main training script. It constructs a CNN architecture inspired by VGGNet and given it's stock settings will evaluate at around 90% on CIFAR-10's designated balanced test batch.

The arguments to provide are:

```--epochs```: The number of epochs you would like to train your model for. Expects an integer value.

```--batch_size```: The batch size for each training iteration. I.e. how many data samples to train at once (more means faster training epochs, less means slower). Expects and integer value

```--val_size```: The percent size you would like the validation size to be. The script will use the same testing data (10,000 images) for evaluation(e.g. the test batch provided in the [CIFAR-10 tech report](https://www.cs.toronto.edu/~kriz/cifar.html)). It takes the other 50,000 images and takes the ```--val_size``` percentage of 50000 as the validation set. I typically run this with about a 0.20 validation size so that the validation size is equal to the test size and it can be used as a good proxy for generalization performance. Expects a floating point value.

```--save_model_path```: The path that you would like to save your model.pb file and other folders for loading later. Expects a path string.

```2. feature_extractor_tSNE.py```:

This script will run an ```sklearn``` t-SNE analysis and provide a plot based on a prior model provided by ```cifar_10_CNN_train.py```
By default, it will extract the penultimate dense later (before the classification layer) for the test data set and use these features as the high-dimensional data sent in to the t-SNE algorithm.

The only argument needed is ```--model_folder``` which will take your pre-trained model and use it on the CIFAR-10 test data for feature extraction.

```3. model_assessment.py```:

This script will take a pre-trained model and return an ROC curve as well as the AUC for the multi-class classification in CIFAR-10. The script will also display a detailed classification report including a multiclass confusion matrix for The CIFAR-10 classes indexed from 0-9.

It requires two arguments ```--model_folder``` where you have your saved model from ```cifar_10_CNN_train.py``` and ```save_plot_path``` which designates where to save the ROC curve produced by the script. 

```4. cifar_10_summary_stats.py```:

This script is used to visualize specific images in the CIFAR-10 batches.

You need to provide ```--batch_id``` which is an integer between 1-6. This corresponds to the batches that are provided when downloading CIFAR-10 and extracting via the python ```pickle``` module. More can be read at the [CIFAR-10 Tech Report](https://www.cs.toronto.edu/~kriz/cifar.html)

You also may provide a ```--sample_id``` to select a specific index of image within each batch. This expects a value between 0-9999 since there are 10000 images in each batch

If you would simply line a 3x3 grid of random images from CIFAR-10 batches. You can specify ```--grid_only```.

Lastly, if you would like to save the image generated you can do so by specifying the ```--save_image``` argument.