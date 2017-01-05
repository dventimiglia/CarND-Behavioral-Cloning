# Imports

from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Input, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pickle

# Load

with open('train.p', 'rb') as f:
    data = pickle.load(f)
with open('./test.p', mode='rb') as f:
    test = pickle.load(f)

# Split

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], random_state=0, test_size=0.33)

# Normalize

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255 - 0.5
X_val = X_val / 255 - 0.5
Y_train = np_utils.to_categorical(y_train, 43)
Y_val = np_utils.to_categorical(y_val, 43)

# Preprocess

X_test = test['features']
y_test = test['labels']
X_test = X_test.astype('float32')
X_test /= 255
X_test -= 0.5
Y_test = np_utils.to_categorical(y_test, 43)

# Model

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))
model.summary()

# Visualize

plot(model, to_file='model.png')

# Train

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val, Y_val))
    
# Test

model.evaluate(X_test, Y_test)

