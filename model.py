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

# Model

input_shape = [160, 320, 3]
input_shape = [80, 160, 3]

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape, trainable=False))
model.add(Conv2D(32, 3, 3))
model.add(MaxPooling2D((2,2)))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.summary()

# Visualize

plot(model, to_file='model.png', show_shapes=True)

# Train

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

def generate_arrays_from_file(index_file, image_base):
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

history = model.fit_generator(generate_arrays_from_file("data/driving_log_head.csv", "data/"), 9, 5)

# Test

model.evaluate(X_test, Y_test)

# Save

model.save("model.h5")
with open("model.json", "w") as f:
    f.write(model.to_json())
