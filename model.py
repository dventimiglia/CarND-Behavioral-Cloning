#!/usr/bin/env python

from PIL import Image
from itertools import groupby, dropwhile, filterfalse, takewhile
from itertools import islice, chain
from itertools import zip_longest, cycle, permutations, combinations, combinations_with_replacement
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Input, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from scipy.stats import kurtosis, skew, describe
from util import process
import cv2
import gc
import keras.preprocessing.image as img
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import random
import sys

# Utilities

def cyclefeed(path):
    while 1:
        f = open(path)
        for line in f:
            yield line
        f.close()

def singlefeed(path):
    f = open(path)
    for line in f:
        yield line
    f.close()

feed = lambda x : [l for l in open(x)]

split = lambda x : (line.split(",") for line in x)

select = lambda x, indices : ([r[i] for i in indices] for r in x)

# process = lambda x, shape : cv2.resize(np.asarray(Image.open(x)), tuple(shape[:-1]), interpolation=cv2.INTER_AREA)

fetch = lambda x, base, shape : ([process(base+f.strip(), shape) for f in record[:1]]+[float(v) for v in record[1:]] for record in x)

flip = lambda y : y if random.choice([True, False]) else [img.flip_axis(y[0],1), -1*y[1]]

weight = lambda y : [y[0], y[1], y[1]*y[1]*0.5 + 0.5]

flip_and_weight = lambda y: weight(flip(y))

group = lambda x, n, fillvalue=None : zip_longest(*([iter(x)]*n), fillvalue=fillvalue)

transpose = lambda x : (list(map(list, zip(*g))) for g in x)

batch = lambda x, indices=[0, 1] : ([np.asarray(t[i]) for i in indices] for t in x)

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
    model.add(Dense(1, activation='tanh', trainable=False, name="Readout"))
    model.compile(loss="mse", optimizer="adam")
    return model

# Data
                     
image_shape = [160, 320, 3]
input_shape = [x//2 for x in image_shape[:2]] + image_shape[2:]
input_shape = [64, 64, 3]
if len(sys.argv)==5:
    training_index = sys.argv[1]
    base_path = sys.argv[2]
    samples_per_epoch = int(sys.argv[3])
    epochs = int(sys.argv[4])
    batch_size = int(sys.arg[5])
else:
    training_index = "data/driving_log_overtrain.csv"
    base_path = "data/"
    samples_per_epoch = 3000
    epochs = 5
    batch_size = 3

# Analyze

# plt.ion()
# print(describe([float(s[1]) for s in select(split(singlefeed("data/driving_log_train.csv")))]))
# print(plt.hist([float(s[1]) for s in select(split(singlefeed("data/driving_log_train.csv")))],bins=100))
# print(describe([l for l in filter(lambda x: math.fabs(x)>0.01, map(lambda x: x*random.choice([1,-1]), [float(l[0]) for l in select(split(singlefeed("data/driving_log_train.csv")),[3])]))]))
# print(plt.hist([l for l in filter(lambda x: math.fabs(x)>0.01, map(lambda x: x*random.choice([1,-1]), [float(l[0]) for l in select(split(singlefeed("data/driving_log_train.csv")),[3])]))],100))

# Train

model = nvidia(input_shape)
model.summary()
plot(model, to_file="model.png", show_shapes=True)

datafeed = select(cycle(fetch(select(split(feed(training_index)), [0,3]), base_path, input_shape)), [0,1])
groupfeed = group(datafeed, batch_size)
batchgen = batch(transpose(groupfeed))
# x, y = zip(*islice(datafeed, 1000))
# x, y = np.asarray(x), np.asarray(y)
history = model.fit_generator(batchgen, samples_per_epoch=samples_per_epoch, nb_epoch=epochs, verbose=2)

# Save

model.save_weights("model.h5")
with open("model.json", "w") as f:
    f.write(model.to_json())

# Cleanup

gc.collect()

# with open("model.json", 'r') as jfile:
#     model = model_from_json(jfile.read())
# model.compile("adam", "mse")
# model.load_weights("model.h5")
