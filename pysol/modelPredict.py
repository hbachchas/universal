import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers import Input
from keras.models import Model
import keras
from my_dataset_loaders import load_test_dataset
from keras.callbacks import CSVLogger
from os import path

print('Imports done')
evaluation_dir = '/home/himanshu/Documents/Projects/GIT/universal/pysol/evaluation'

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
print('Loading dataset...')
X_test, y_test = load_test_dataset()
print('X_test = ', X_test.shape)
print('y_test = ', y_test.shape)
print('Dataset loaded.')

# normalize inputs from 0-255 to 0.0-1.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

# one hot encode outputs
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]    # get number of classes
