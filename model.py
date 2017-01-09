#!/usr/bin/env python

# Imports

from PIL import Image
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Input, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.visualize_util import plot
import cv2
import gc
import keras.preprocessing.image as img
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import sys

# Utilities

def input_generator(index_file, image_base, target_shape):
    while 1:
        f = open(index_file)
        for line in f:
            center, left, right, angle, throttle, brake, speed = line.split(",")
            img = center
            img = image_base + img.strip()
            img = np.asarray(Image.open(img))
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
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(6, 3, 3, name="Conv2D1", activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2,2), name="MaxPool1"))
    model.add(Conv2D(10, 3, 3, activation="relu", name="Conv2D2"))
    model.add(MaxPooling2D((2,2), name="MaxPool2"))
    model.add(Conv2D(16, 3, 3, activation="relu", name="Conv2D3"))
    model.add(MaxPooling2D((2,2), name="MaxPool3"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(128, activation="relu", name="FC1"))
    model.add(Dense(64, activation="relu", name="FC2"))
    model.add(Dense(32, activation="relu", name="FC3"))
    model.add(Dense(16, activation="relu", name="FC4"))
    model.add(Dense(8, activation="relu", name="FC5"))
    model.add((Dropout(0.5, name="Dropout")))
    model.add(Dense(1, activation="sigmoid", name="Readout"))
    model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
    return model

def Nvidia(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), name="Conv2D1", activation='relu', input_shape=input_shape))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), name="Conv2D2", activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), name="Conv2D3", activation='relu'))
    model.add(Conv2D(64, 3, 3, name="Conv2D4", activation='relu'))
    model.add(Conv2D(64, 3, 3, name="Conv2D5", activation='relu'))
    model.add(Conv2D(64, 3, 3, name="Conv2D6", activation='relu'))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(128, activation='relu', name="FC1"))
    model.add(Dense(64, activation='relu', name="FC2"))
    model.add(Dense(32, activation='relu', name="FC3"))
    model.add(Dense(16, activation="relu", name="FC4"))
    model.add(Dense(8, activation="relu", name="FC5"))
    model.add(Dense(1, activation="sigmoid", name="Readout"))
    model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
    return model
                     
model = LeNet(input_shape)
model.summary()

# Visualize

plot(model, to_file="model.png", show_shapes=True)

# Train

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

print(sys.argv)

datagen = input_generator(sys.argv[1], sys.argv[2], input_shape)
history = model.fit_generator(datagen, samples_per_epoch=int(sys.argv[3]), nb_epoch=int(sys.argv[4]), verbose=2)

# Cleanup

gc.collect()
