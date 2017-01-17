#!/usr/bin/env python

from PIL import Image
from itertools import groupby, islice, zip_longest, cycle
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Lambda
from keras.layers.convolutional import Cropping2D
from keras.models import Sequential, model_from_json
from keras.utils.visualize_util import plot
from scipy.stats import kurtosis, skew, describe
import cv2
import gc
import keras.preprocessing.image as img
import math
import numpy as np
import os
import random
import sys


# Utilities

def rcycle(iterable):
    """Return elements from the iterable.  Shuffle the elements of the
iterable when it becomes exhausted, then begin returning them
again.  Repeat this sequence of operations indefinitely.  Note
that the elements of the iterable are essentially returned in
batches, and that the first batch is not shuffled.  If you want
only to return random elements then you must know batch size,
which will be the number of elements in the underlying finite
iterable, and you must discard the first batch.  The
itertools.islice function can be helpful here.
    """
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
              yield element
              

def feed(filename):
    """Return an iterable over the lines the file with name
'filename'."""
    return (l for l in open(filename))


def split(lines, delimiter=","):
    """Return an iterable over 'lines', splitting each into records
comprising fields, using 'delimiter'."""
    return (line.split(delimiter) for line in lines)


def select(fields, indices):
    """Return an iterable over records of fields, selecting only the
fields listed in 'indices'."""
    return ([r[i] for i in indices] for r in fields)


def load(f):
    """Return a NumPy array for the image indicated by 'f'."""
    return np.asarray(Image.open(f))


def fetch(records, base):
    """Return an iterable over 'records', fetching a sample [X,y] for each
record.  A sample is a ordered pair with the first element X a
NumPy array (typically, an image) and the second element y
floating point number (the label)."""
    return ([load(base+f.strip()) for f in record[:1]]+[float(v) for v in record[1:]] for record in records)


def rflip(x):
    """Randomly flip an image 'x' along its horizontal axis 50% of the
    time."""
    return x if random.choice([True, False]) else [img.flip_axis(x[0],1), -1*x[1]]


def rshift(x, factor=0.1):
    """Randomly shift an image 'x' along its horizontal axis by 'factor'
amount."""
    return img.random_shift(x, factor, 0.0, 0, 1, 2, fill_mode='wrap')


def crop(x, shape):
    """Crop an image 'x' by the boundaries given in 'shape', which is a
tuple of tuples: ((x1,x2),(y1,y2))."""
    return x[shape[0][0]:shape[0][1],shape[1][0]:shape[1][1]]


def resize(x, shape):
    """Resize an image 'x' in its height and width dimensions (not in its
channel dimension) according to 'shape'.  Note that in this case,
'shape' is a triple such as is returned by the NumPy 'shape'
operation."""
    return cv2.resize(x, tuple(shape[:2]))


def process(data, crop_shape, resize_shape):
    """Process 'data' into a cropped and resized NumPy array image."""
    return resize(crop(data, crop_shape), resize_shape)


def group(items, n, fillvalue=None):
    """Iterate over 'items' but return them in groups of size 'n'.  If
need be, fill the last group with 'fillvalue'."""
    return zip_longest(*([iter(items)]*n), fillvalue=fillvalue)


def transpose(tuples):
    """Transpose items in the 'tuples' iterable.  Each item is expected to
be a tuple and all tuples are expected to have the same length.  The
transposition is such that for each position 'i' within the tuples,
all of the elements at that position across all the items are
themselves grouped together.  Each group is realized into a list.  If
'tuples' contains m items and each item is itself a tuple of n
elements, then what is returned is a set of n lists, and each list
contains m elements.  The n lists are themselves presented as an
iterable."""
    return (list(map(list, zip(*g))) for g in tuples)


def batch(groups, indices=[0, 1]):
    """Iterate over 'groups', each of which is itself an iterable (such as
a list), and turn the groups selected by 'indices' into a NumPy array.
Naturally, the groups are expected to be of items that are compatible
with NumPy arrays, which would be any of the appropriate numeric
types."""
    return ([np.asarray(t[i]) for i in indices] for t in groups)


# Model
    

def CarND(input_shape):
    """Return a Keras neural network model."""
    model = Sequential()

    # Normalize input.
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name="Normalize"))

    # Reduce dimensions through trainable convolution, activation, and
    # pooling layers.
    model.add(Conv2D(24, 5, 5, subsample=(1,1), name="Conv2D1", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool1"))
    model.add(Conv2D(36, 5, 5, subsample=(1,1), name="Conv2D2", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool2"))
    model.add(Conv2D(48, 5, 5, subsample=(1,1), name="Conv2D3", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool3"))

    # Flatten input in a non-trainable layer before feeding into
    # fully-connected layers.
    model.add(Flatten(name="Flatten"))

    # Model steering through trainable layers comprising dense units
    # as ell as dropout units for regularization.
    model.add(Dense(100, activation="relu", name="FC2"))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu", name="FC3"))
    model.add(Dense(10, activation="relu", name="FC4"))

    # Generate output (steering angles) with a single non-trainable
    # node.
    model.add(Dense(1, name="Readout", trainable=False))
    return model


# Training

def pipeline(theta, training=False):
    """Create a data-processing pipeline.  The 'training_index' parameter
is the name of a CSV index file specifying samples, with fields for
image filenames and for steering angles.  The 'base_path' parameter is
the directory path for the image filenames.  The pipeline itself is a
generator (which is an iterable), where each item from the generator
is a batch of samples (X,y).  X and y are each NumPy arrays, with X as
a batch of images and y as a batch of outputs.  The images in X are
cropped and reshaped according to the 'resize_shape' and 'crop_shape'
parameters.  Finally, augmentation may be performed if a training
pipeline is desired, determined by the 'training' parameter.  Training
pipelines have their images randomly flipped along the horizontal
axis, and are randomly shifted along their horizontal axis."""
    samples = select(rcycle(fetch(select(split(feed(theta.training_index)), [0,3]), theta.base_path)), [0,1])
    if training:
        if theta.flip:
            samples = (rflip(x) for x in samples)
        if theta.shift:
            samples = ((rshift(x[0]),x[1]) for x in samples)
    samples = ((process(x[0], theta.crop_shape, theta.resize_shape), x[1]) for x in samples)
    groups = group(samples, theta.batch_size)
    batches = batch(transpose(groups))
    return batches


def train(model):
    """Train the model."""
    traingen = pipeline(theta, training=True)
    validgen = pipeline(theta)
    history = model.fit_generator(
        traingen,
        theta.samples_per_epoch,
        theta.epochs,
        validation_data=validgen,
        nb_val_samples=theta.valid_samples_per_epoch)


# Data Structures

class HyperParameters:
    """Essentially a struct just to gather hyper-parameters into one
place, for convenience."""
    def __init__(self):
        return


# Entry-point

if __name__=="__main__":        # In case this module is imported
    theta = HyperParameters()
    theta.crop_shape = ((70,140),(0,320))
    theta.resize_shape = [64, 64, 3]
    theta.samples_per_epoch = 3000
    theta.valid_samples_per_epoch = 3000
    theta.epochs = 5
    theta.batch_size = 100
    theta.training_index = "data/driving_log_overtrain.csv"
    theta.validation_index = "data/driving_log_overtrain.csv"
    theta.base_path = "data/"
    theta.flip = False
    theta.shift = False
    if len(sys.argv)>0:         # Running from the command line
        theta.training_index = os.environ['TRAINING_INDEX']
        theta.validation_index = os.environ['VALIDATION_INDEX']
        theta.base_path = os.environ['BASE_PATH']
        theta.samples_per_epoch = int(os.environ['SAMPLES_PER_EPOCH'])
        theta.valid_samples_per_epoch = int(os.environ['VALID_SAMPLES_PER_EPOCH'])
        theta.epochs = int(os.environ['EPOCHS'])
        theta.batch_size = int(os.environ['BATCH_SIZE'])
        theta.flip = os.environ['FLIP']=='yes'
        theta.shift = os.environ['SHIFT']=='yes'
    model = CarND([theta.resize_shape[1],
                   theta.resize_shape[0],
                   theta.resize_shape[2]])
    model.compile(loss="mse", optimizer="adam")
    model.theta = theta
    model.summary()
    plot(model, to_file="model.png", show_shapes=True)
    print(theta.__dict__)
    train(model)
    model.save_weights("model.h5")
    with open("model.json", "w") as f:
        f.write(model.to_json())
    gc.collect()
