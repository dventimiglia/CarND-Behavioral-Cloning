# Imports

from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Input, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
import cv2
import keras.preprocessing.image as img
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

# Utilities

def input_generator(index_file, image_base, target_shape):
    while 1:
        f = open(index_file)
        for line in f:
            center, left, right, angle, throttle, brake, speed = line.split(",")
            img = center
            img = image_base + img.strip()
            img = plt.imread(img)
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation = cv2.INTER_AREA)
            img = np.reshape(img, (1,)+img.shape)
            angle = np.array([[float(angle)]])
            yield (img, angle)
        f.close()

# Model

image_shape = [160, 320, 3]
input_shape = [x//2 for x in image_shape[:2]] + image_shape[2:]

def LeNet(input_shape):
    model = Sequential()
    # Scale image values to [-1, 1]
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(6, 3, 3, name="Conv2D1", input_shape=input_shape))
    model.add(Activation("relu", name="Activation1"))
    model.add(MaxPooling2D((2,2), name="MaxPool1"))
    model.add(Conv2D(10, 3, 3, name="Conv2D2"))
    model.add(Activation("relu", name="Activation2"))
    model.add(MaxPooling2D((2,2), name="MaxPool2"))
    model.add(Conv2D(16, 3, 3, name="Conv2D3"))
    model.add(Activation("relu", name="Activation3"))
    model.add(MaxPooling2D((2,2), name="MaxPool3"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(128, activation="relu", name="Fully-Connected1"))
    model.add(Dense(64, activation="relu", name="Fully-Connected2"))
    model.add(Dense(32, activation="relu", name="Fully-Connected3"))
    model.add(Dense(16, activation="relu", name="Fully-Connected4"))
    model.add(Dense(8, activation="relu", name="Fully-Connected5"))
    model.add((Dropout(0.5, name="Dropout")))
    model.add(Dense(1, activation="sigmoid", name="Readout"))
    model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
    return model

model = LeNet(input_shape)
model.summary()

# Visualize

plot(model, to_file="model.png", show_shapes=True)

# Train

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
datagen = input_generator("data/driving_log_three.csv", "data/", input_shape)
history = model.fit_generator(datagen, samples_per_epoch=3, nb_epoch=20, verbose=2)

for i in range(3):
    a = datagen.__next__()
    x = a[0]
    y = a[1]
    print(model.predict(x), y)
