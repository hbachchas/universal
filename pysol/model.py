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
from my_dataset_loaders import load_dataset
from keras.callbacks import CSVLogger
from os import path

print('Imports done')
evaluation_dir = '/home/himanshu/Documents/Projects/GIT/universal/pysol/evaluation'

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
print('Loading dataset...')
X_train, y_train, X_test, y_test = load_dataset()
print('Dataset loaded.')

# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]    # get number of classes

# Define the model
model = Sequential()
# model.add(Conv2D(48, (3, 3), input_shape=(144, 144, 3), padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(48, (3, 3), input_shape=(144, 144, 3), padding='same'))
model.add(BatchNormalization())    # using batch_norm, without activation, it works best on linear inputs
model.add(Activation('relu'))      # apply batch_norm before non-linearity
# model.add(Dropout(0.2))       # ignored because we want to observe simple behavior
model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())    # using batch_norm, without activation, it works best on linear inputs
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 32
lrate = 0.01
decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('Model compiled.')

# Fit the model
print('Fitting the model...')
csv_logger = CSVLogger(path.join(evaluation_dir,'log.csv'), append=True, separator=';')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=1, callbacks=[csv_logger])

# Save model architecture and weights
# serialize model to JSON
model_json = model.to_json()
with open(path.join(evaluation_dir,'model.json'), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path.join(evaluation_dir,'model.h5'))
print("Model saved to disk")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Visualize the training progress     # keep verbose=1 in model.fit()
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig(path.join(evaluation_dir,'Acc.eps'), format='eps', dpi=400)
# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
# plt.show()
plt.savefig(path.join(evaluation_dir,'Loss.eps'), format='eps', dpi=400)
print('Plots saved to disk')
