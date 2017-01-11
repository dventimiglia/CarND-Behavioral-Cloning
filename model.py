#!/usr/bin/env python

from PIL import Image
from itertools import groupby
from itertools import islice, chain
from itertools import zip_longest, cycle, permutations, combinations, combinations_with_replacement
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

def cyclefeed(path):
    while 1:
        f = open(path)
        for line in f:
            yield line
        f.close()

feed = lambda x : [l for l in open(x)]

split = lambda x : (line.split(",") for line in x)

select = lambda x, indices=[0, 3]: ([r[i] for i in indices] for r in x)

fetch = lambda x, base, shape : ([cv2.resize(np.asarray(img.load_img(base+f.strip())), shape, interpolation=cv2.INTER_AREA) for f in record[:1]]+[float(v) for v in record[1:]] for record in x)

pair = lambda x, l, r : ([s[l], s[r]] for s in x)

group = lambda x, n, fillvalue=None : zip_longest(*([iter(x)]*n), fillvalue=fillvalue)

transpose = lambda x : (list(map(list, zip(*g))) for g in x)

batch = lambda x : ([np.asarray(t[0]), np.asarray(t[1])] for t in x)

# Model

def nvidia(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), name="Conv2D1", activation='relu', input_shape=input_shape))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), name="Conv2D2", activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), name="Conv2D3", activation='relu'))
    model.add(Conv2D(64, 5, 5, name="Conv2D4", activation='relu'))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(1164, activation='relu', name="FC1"))
    model.add(Dense(100, activation='relu', name="FC2"))
    model.add(Dense(50, activation='relu', name="FC3"))
    model.add(Dense(10, activation='relu', name="FC4"))
    model.add(Dense(1, activation='sigmoid', name="Readout"))
    model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
    model.compile(loss="mse", optimizer="adam")
    return model

# Training
                     
image_shape = [160, 320, 3]
input_shape = [x//2 for x in image_shape[:2]] + image_shape[2:]
input_shape = [64, 64, 3]
training_index = sys.argv[1]
base_path = sys.argv[2]
samples = sys.argv[3]
epochs = sys.argv[4]

model = nvidia(input_shape)
model.summary()
plot(model, to_file="model.png", show_shapes=True)

training = batch(transpose(group(pair(cycle(fetch(select(split(feed(training_index))), base_path, (input_shape[1], input_shape[0]))), 0, 1), 100)))
history = model.fit_generator(training, samples_per_epoch=samples, nb_epoch=epochs, verbose=2)

# Save

model.save_weights("model.h5")
with open("model.json", "w") as f:
    f.write(model.to_json())

# Cleanup

gc.collect()
