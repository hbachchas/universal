import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers import Input
from keras.models import Model
import keras
from my_dataset_loaders import load_dataset_RGB
from keras.callbacks import CSVLogger

print('Imports done')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
print('Loading dataset...')
X1_train, X2_train, X3_train, y_train, X1_test, X2_test, X3_test, y_test = load_dataset_RGB()
print('Dataset loaded.')

# Define the vision modules
s_input = Input(shape=(96, 128, 3))
# x = Conv2D(64, (3, 3), padding='same', kernel_constraint=maxnorm(3))(s_input)
x = Conv2D(64, (3, 3), padding='same', kernel_constraint=maxnorm(3))(s_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(x)
out = Dropout(0.5)(x)
print('Model defined.')

vision_model = Model(s_input, out)

# Then define the input-apart model
s_a = Input(shape=(96, 128, 3))
s_b = Input(shape=(96, 128, 3))
s_c = Input(shape=(96, 128, 3))

# The vision model will be shared
out_a = vision_model(s_a)
out_b = vision_model(s_b)
out_c = vision_model(s_c)

concatenated = keras.layers.concatenate([out_a, out_b, out_c])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([s_a, s_b, s_c], out)

# Compile model
epochs = 30
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classification_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(classification_model.summary())
print('Model compiled.')

# Fit the model
print('Fitting the model...')
csv_logger = CSVLogger('./evaluation/log.csv', append=True, separator=';')
history = classification_model.fit([X1_train, X2_train, X3_train], y_train, validation_data=([X1_test, X2_test, X3_test], y_test), epochs=epochs, batch_size=32, verbose=1, callbacks=[csv_logger])

# Save model architecture and weights
# serialize model to JSON
model_json = classification_model.to_json()
with open("./evaluation/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classification_model.save_weights("./evaluation/model.h5")
print("Model saved to disk")

# Final evaluation of the model
scores = classification_model.evaluate([X1_test, X2_test, X3_test], y_test, verbose=0)
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
plt.savefig('./evaluation/Acc.eps', format='eps', dpi=400)
# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
# plt.show()
plt.savefig('./evaluation/Loss.eps', format='eps', dpi=400)
print('Plots saved to disk')
