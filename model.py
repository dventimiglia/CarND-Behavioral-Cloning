# Imports

from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Input, Dropout, Lambda
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

def input_generator(index_file, image_base):
    while 1:
        f = open(index_file)
        next(f)
        for line in f:
            center, left, right, angle, throttle, brake, speed = line.split(",")
            center = image_base + center.strip()
            left = image_base + left.strip()
            right = image_base + right.strip()
            center = cv2.imread(center)
            center = cv2.resize(center, (input_shape[1], input_shape[0]), interpolation = cv2.INTER_AREA)
            left = cv2.imread(left)
            right = cv2.imread(right)
            angle = float(angle)
            throttle = float(throttle)
            brake = float(brake)
            speed = float(speed)
            batch_x = np.zeros((1,) + center.shape)
            batch_y = np.zeros([1,1])
            batch_x[0] = center
            batch_y[0] = angle
            yield (batch_x, batch_y)
        f.close()

# Model

image_shape = [160, 320, 3]
input_shape = [x//2 for x in image_shape[:2]] + image_shape[2:]

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False, name="Preprocess"))
model.add(Conv2D(32, 3, 3, name="Conv2d"))
model.add(MaxPooling2D((2,2), name="MaxPool"))
model.add((Dropout(0.5, name="Dropout")))
model.add(Activation("relu", name="Activation"))
model.add(Flatten(name="Flatten"))
model.add(Dense(128, activation="relu", name="Fully-Connected"))
model.add(Dense(1, activation="sigmoid", name="Readout"))
model.add(Lambda(lambda x: 2.*x-1., trainable=False, name="Postprocess"))
model.summary()

# Visualize

plot(model, to_file="model.png", show_shapes=True)

# Train

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

datagen = input_generator("data/driving_log_head.csv", "data/")
history = model.fit_generator(datagen, 9, 5, verbose=2)

# Test

# model.evaluate(X_test, Y_test)

# Save

# model.save("model.h5")
# with open("model.json", "w") as f:
#     f.write(model.to_json())

