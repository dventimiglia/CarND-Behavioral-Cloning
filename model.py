#!/usr/bin/env python

# Imports

from PIL import Image
from itertools import groupby
from itertools import islice, chain
from itertools import zip_longest
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

def lines(path):
    while 1:
        f = open(path)
        for line in f:
            yield line
        f.close()

records = lambda x : (line.split(",") for line in x)

samples = lambda x, base, shape : ([cv2.resize(np.asarray(Image.open(base+f.strip())), shape, interpolation=cv2.INTER_AREA) for f in record[0:3]]+[float(v) for v in record[3:]] for record in x)

pairs = lambda x, l, r : ([s[l], s[r]] for s in x)

groups = lambda x, n, fillvalue=None : zip_longest(*([iter(x)]*n), fillvalue=fillvalue)

transpositions = lambda x : (list(map(list, zip(*g))) for g in x)

batches = lambda x : ([np.asarray(t[0]), np.asarray(t[1])] for t in x)

# Model

image_shape = [160, 320, 3]
input_shape = [x//2 for x in image_shape[:2]] + image_shape[2:]

def lenet(input_shape):
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

def nvidia(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), name="Conv2D1", activation='relu', input_shape=input_shape))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), name="Conv2D2", activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), name="Conv2D3", activation='relu'))
    model.add(Conv2D(64, 5, 5, name="Conv2D4", activation='relu'))
    model.add(Conv2D(64, 3, 3, name="Conv2D5", activation='relu'))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(1164, activation='relu', name="FC1"))
    model.add(Dense(100, activation='relu', name="FC2"))
    model.add(Dense(50, activation='relu', name="FC3"))
    model.add(Dense(10, activation="relu", name="FC4"))
    model.add(Dense(1, activation="sigmoid", name="Readout"))
    model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
    return model
                     
model = nvidia(input_shape)
model.summary()

# Visualize

plot(model, to_file="model.png", show_shapes=True)

# Train

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

print(sys.argv)

# index_file = "data/driving_log_train.csv"
# index_file = "data/driving_log_random_sample.csv"
index_file = "data/driving_log_train.csv"
base_path = "data/" 

generator = batches(transpositions(groups(pairs(samples(records(lines(index_file)), base_path, (input_shape[1], input_shape[0])), 0, 3), 32)))
history = model.fit_generator(generator, samples_per_epoch=1000, nb_epoch=10, verbose=2)

# Save

model.save("model.h5")
with open("model.json", "w") as f:
    f.write(model.to_json())

# Cleanup

gc.collect()
