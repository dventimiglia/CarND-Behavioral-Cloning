import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math

# TODO: Implement load the data here.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Use `train_test_split` here.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], random_state=0, test_size=0.33)

# TODO: Implement data normalization here.
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255 - 0.5
X_val = X_val / 255 - 0.5

# TODO: Build a two-layer feedforward neural network with Keras here.
from keras.models import Sequential
from keras.layers import Dense, Input, Activation

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))
model.add(Dense(43, activation='softmax'))

# TODO: Compile and train the model here.
from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train, 43)
Y_val = np_utils.to_categorical(y_val, 43)

X_train_flat = X_train.reshape(-1, 32*32*3)
X_val_flat = X_val.reshape(-1, 32*32*3)

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train_flat, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val_flat, Y_val))

# TODO: Re-construct the network and add a convolutional layer before the first fully-connected layer.
# NOTE: RELOAD DATA & NORMALIZE BEFORE RUNNING 
from keras.layers import Conv2D, Flatten

Y_train = np_utils.to_categorical(y_train, 43)
Y_val = np_utils.to_categorical(y_val, 43)


model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val, Y_val))

# TODO: Re-construct the network and add a pooling layer after the convolutional layer.
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation

Y_train = np_utils.to_categorical(y_train, 43)
Y_val = np_utils.to_categorical(y_val, 43)


model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val, Y_val))

# TODO: Re-construct the network and add dropout after the pooling layer.
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val, Y_val))

# TODO: Load test data
with open('./test.p', mode='rb') as f:
    test = pickle.load(f)
    
# TODO: Preprocess data & one-hot encode the labels
X_test = test['features']
y_test = test['labels']
X_test = X_test.astype('float32')
X_test /= 255
X_test -= 0.5
Y_test = np_utils.to_categorical(y_test, 43)

# TODO: Evaluate model on test data
model.evaluate(X_test, Y_test)
