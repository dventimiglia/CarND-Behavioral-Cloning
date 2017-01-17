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
from random import shuffle
from scipy.stats import kurtosis, skew, describe
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

plt.ion()

# Utilities

def rcycle(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        shuffle(saved)
        for element in saved:
              yield element
              
feed = lambda x : (l for l in open(x))

split = lambda x : (line.split(",") for line in x)

select = lambda x, indices: ([r[i] for i in indices] for r in x)

load = lambda x: np.asarray(Image.open(x))

fetch = lambda x, base: ([load(base+f.strip()) for f in record[:1]]+[float(v) for v in record[1:]] for record in x)

flip = lambda x: x if random.choice([True, False]) else [img.flip_axis(x[0],1), -1*x[1]]

shift = lambda x: img.random_shift(x, 0.1, 0.0, 0, 1, 2, fill_mode='wrap')

crop = lambda x, s: x[s[0][0]:s[0][1],s[1][0]:s[1][1]]

resize = lambda x, s: cv2.resize(x, tuple(s[:2]))

process = lambda x, c, s: resize(crop(x, c), s)

group = lambda x, n, fillvalue=None: zip_longest(*([iter(x)]*n), fillvalue=fillvalue)

transpose = lambda x: (list(map(list, zip(*g))) for g in x)

batch = lambda x, indices=[0, 1]: ([np.asarray(t[i]) for i in indices] for t in x)

# Model

def dventimi(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
    model.add(Conv2D(24, 5, 5, subsample=(1,1), name="Conv2D1", activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(name="MaxPool1"))
    model.add(Conv2D(36, 5, 5, subsample=(1,1), name="Conv2D2", activation='relu'))
    model.add(MaxPooling2D(name="MaxPool2"))
    model.add(Conv2D(48, 5, 5, subsample=(1,1), name="Conv2D3", activation='relu'))
    model.add(MaxPooling2D(name="MaxPool3"))
    model.add(Conv2D(64, 3, 3, name="Conv2D4", activation='relu'))
    model.add(MaxPooling2D(name="MaxPool4"))
    model.add(Flatten(name="Flatten"))
    model.add(Dense(100, activation='relu', name="FC2"))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', name="FC3"))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', name="FC4"))
    # model.add(Dropout(0.5))
    model.add(Dense(1, name="Readout", trainable=False))
    model.compile(loss="mse", optimizer="adam")
    return model

# Analyze

# def analyze():
#     plt.ion()
#     print(describe([float(s[1]) for s in select(split(singlefeed("data/driving_log_train.csv")))]))
#     print(plt.hist([float(s[1]) for s in select(split(singlefeed("data/driving_log_train.csv")))],bins=100))
#     print(describe([l for l in filter(lambda x: math.fabs(x)>0.01, map(lambda x: x*random.choice([1,-1]), [float(l[0]) for l in select(split(singlefeed("data/driving_log_train.csv")),[3])]))]))
#     print(plt.hist([l for l in filter(lambda x: math.fabs(x)>0.01, map(lambda x: x*random.choice([1,-1]), [float(l[0]) for l in select(split(singlefeed("data/driving_log_train.csv")),[3])]))],100))

# Train

def pipeline(training_index, base_path, input_shape, crop_shape, training=False):
    samples = select(rcycle(fetch(select(split(feed(training_index)), [0,3]), base_path)), [0,1])
    if training:
        samples = (flip(x) for x in samples)
        # samples = ((shift(x[0]),x[1]) for x in samples)
    # samples = ((resize(crop(x[0], crop_shape), input_shape), x[1]) for x in samples)
    samples = ((process(x[0], crop_shape, input_shape), x[1]) for x in samples)
    groups = group(samples, batch_size)
    batches = batch(transpose(groups))
    return batches

def train():
    traingen = pipeline(training_index, base_path, input_shape, crop_shape, training=True)
    validgen = pipeline(validation_index, base_path, input_shape, crop_shape)
    history = model.fit_generator(traingen, samples_per_epoch, epochs, validation_data=validgen, nb_val_samples=valid_samples_per_epoch)

if __name__=="__main__":
    if len(sys.argv)>1:
        training_index = sys.argv[1]
        validation_index = sys.argv[2]
        base_path = sys.argv[3]
        samples_per_epoch = int(sys.argv[4])
        valid_samples_per_epoch = int(sys.argv[4])
        epochs = int(sys.argv[5])
        batch_size = int(sys.argv[6])
    else:
        training_index = "data/driving_log_overtrain.csv"
        validation_index = "data/driving_log_overtrain.csv"
        base_path = "data/"
        samples_per_epoch = 3000
        valid_samples_per_epoch = 3000
        epochs = 5
        batch_size = 100
    # input_shape = [256, 60, 3]
    input_shape = [64, 64, 3]
    crop_shape = ((100,140),(0,320))
    model = dventimi([input_shape[1],input_shape[0],input_shape[2]])
    model.summary()
    plot(model, to_file="model.png", show_shapes=True)
    train()
    model.save_weights("model.h5")
    with open("model.json", "w") as f:
        f.write(model.to_json())
    gc.collect()

# lines = open(training_index)
# records = (l.split(",") for l in lines)
# columns = zip(*((process(base_path + r[0].strip(), input_shape), float(r[3])) for r in records))
# X_train, y_train = (np.asarray(c) for c in columns)
# X_train = np.append(X_train, X_train[:,:,::-1], axis=0)
# y_train = np.append(y_train, -y_train, axis=0)
# history = model.fit(X_train, y_train, batch_size, epochs)
