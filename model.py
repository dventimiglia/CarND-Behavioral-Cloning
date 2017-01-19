
# Utilities

#       Return elements from the iterable.  Shuffle the elements of the
#       iterable when it becomes exhausted, then begin returning them
#       again.  Repeat this sequence of operations indefinitely.  Note
#       that the elements of the iterable are essentially returned in
#       batches, and that the first batch is not shuffled.  If you want
#       only to return random elements then you must know batch size,
#       which will be the number of elements in the underlying finite
#       iterable, and you must discard the first batch.  The
#       itertools.islice function can be helpful here.

from PIL import Image
from itertools import groupby, islice, zip_longest, cycle, filterfalse
from scipy.stats import kurtosis, skew, describe
import keras.preprocessing.image as img
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# #+RESULTS:

def rcycle(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
              yield element

# #+RESULTS:

feed = lambda filename: (l for l in open(filename))
split = lambda lines, delimiter=",": (line.split(delimiter) for line in lines)
select = lambda fields, indices: ([r[i] for i in indices] for r in fields)
load = lambda f: np.asarray(Image.open(f))
fetch = lambda records, base: ([load(base+f.strip()) for f in record[:1]]+[float(v) for v in record[1:]] for record in records)
rshift = lambda x, factor=0.1: img.random_shift(x, factor, 0.0, 0, 1, 2, fill_mode='wrap')
group = lambda items, n, fillvalue=None: zip_longest(*([iter(items)]*n), fillvalue=fillvalue)
transpose = lambda tuples: (list(map(list, zip(*g))) for g in tuples)
batch = lambda groups, indices=[0, 1]: ([np.asarray(t[i]) for i in indices] for t in groups)

# Exploratory Analysis

#       Get the target labels---the steering angles---for /all/ of the
#       data, from =data/driving_log_all.csv=, plot a histogram, and
#       generate basic descriptive statistics.

f = plt.figure()
y1 = np.array([float(s[0]) for s in select(split(feed("data/driving_log_all.csv")),[3])])
h = plt.hist(y1,bins=100)
s = plt.savefig("hist1.png", format='png')
describe(y1)

# #+RESULTS:
#       : DescribeResult(nobs=8036, minmax=(-0.94269539999999996, 1.0), mean=0.0040696440648332515, variance=0.016599764281272529, skewness=-0.13028924577521922, kurtosis=6.311554102057668)

#       #+CAPTION: All Samples - No Reflection
#       #+ATTR_HTML: :alt CarND/Architecture Image :title Architecture
#       [[file:hist1.png]]

#       The data have non-zero /mean/ and /skewness/.  perhaps arising
#       from a bias toward left-hand turns when driving on a closed
#       track.

#       - mean=0.0040696440648332515
#       - skewness=-0.13028924577521922

#       The data are dominated by small steering angles because the car
#       spends most of its time on the track in straightaways.  The
#       asymmetry in the data is more apparent if we mask out small
#       angles and repeat the analysis.  Steering angles occupy the
#       interval [-1, 1], but the "straight" samples appear to be within
#       the neighborhood [-0.01, 0.01].

#       We might consider masking out small angled samples from the
#       actual training data as well, a subject we shall return to in a
#       later section.

f = plt.figure()
p = lambda x: abs(x)<0.01
y2 = np.array([s for s in filterfalse(p,y1)])
h = plt.hist(y2,bins=100)
s = plt.savefig("hist2.png", format='png')
describe(y2)

# #+RESULTS:
#       : DescribeResult(nobs=8036, minmax=(-0.94269539999999996, 1.0), mean=0.0040696440648332515, variance=0.016599764281272529, skewness=-0.13028924577521922, kurtosis=6.311554102057668)

#       #+CAPTION: abs(angle)>0.01 - No Reflection
#       #+ATTR_HTML: :alt CarND/Architecture Image :title Architecture
#       [[file:hist2.png]]

#       A simple trick that we can play to remove this asymmetry---if we
#       wish---is to join the data with its reflection, effectively
#       doubling our sample size in the process.  For illustration
#       purposes only, we shall again mask out small angle samples.

f = plt.figure()
y3 = np.append(y2, -y2)
h = plt.hist(y3,bins=100)
s = plt.savefig("hist3.png", format='png')
describe(y3)

# Model

#       - Crop :: crop to region (/non-trainable/)
#       - Resize :: reduce scale (/non-trainable/)
#       - Normalize :: scale values to [-1, 1] (/non-trainable/)
#       - Convolution :: learn spatial features and compress
#       - MaxPool :: reduce model size
#       - Dropout :: add regularization (/non-trainable/)
#       - Flatten :: stage to fully-connected layers (/non-trainable/)
#       - FC :: fully-connected layers
#       - Readout :: single node steering angle (/non-trainable/)

#       Return a Keras neural network model.

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Lambda, AveragePooling2D
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.models import Sequential, model_from_json
from keras.utils.visualize_util import plot

def CarND(input_shape):
    model = Sequential()
 
    # Crop
    model.add(Cropping2D(((80,20),(1,1)), input_shape=input_shape, name="Crop"))
 
    # Resize
    model.add(AveragePooling2D(pool_size=(1,4), name="Resize", trainable=False))
 
    # Normalize input.
    model.add(Lambda(lambda x: x/127.5 - 1., name="Normalize"))
 
    # Reduce dimensions through trainable convolution, activation, and
    # pooling layers.
    model.add(Convolution2D(24, 3, 3, subsample=(2,2), name="Convolution2D1", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool1"))
    model.add(Convolution2D(36, 3, 3, subsample=(1,1), name="Convolution2D2", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool2"))
    model.add(Convolution2D(48, 3, 3, subsample=(1,1), name="Convolution2D3", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool3"))
 
    # Dropout for regularization
    model.add(Dropout(0.1, name="Dropout"))
 
    # Flatten input in a non-trainable layer before feeding into
    # fully-connected layers.
    model.add(Flatten(name="Flatten"))
 
    # Model steering through trainable layers comprising dense units
    # as ell as dropout units for regularization.
    model.add(Dense(100, activation="relu", name="FC2"))
    model.add(Dense(50, activation="relu", name="FC3"))
    model.add(Dense(10, activation="relu", name="FC4"))
 
    # Generate output (steering angles) with a single non-trainable
    # node.
    model.add(Dense(1, name="Readout", trainable=False))
    return model

# #+RESULTS:

CarND([160, 320, 3]).summary()

# #+RESULTS:
#       #+begin_example
#       ____________________________________________________________________________________________________
#       Layer (type)                     Output Shape          Param #     Connected to                     
#       ====================================================================================================
#       Crop (Cropping2D)                (None, 60, 318, 3)    0           cropping2d_input_10[0][0]        
#       ____________________________________________________________________________________________________
#       Resize (AveragePooling2D)        (None, 60, 79, 3)     0           Crop[0][0]                       
#       ____________________________________________________________________________________________________
#       Normalize (Lambda)               (None, 60, 79, 3)     0           Resize[0][0]                     
#       ____________________________________________________________________________________________________
#       Convolution2D1 (Convolution2D)   (None, 29, 39, 24)    672         Normalize[0][0]                  
#       ____________________________________________________________________________________________________
#       MaxPool1 (MaxPooling2D)          (None, 14, 19, 24)    0           Convolution2D1[0][0]             
#       ____________________________________________________________________________________________________
#       Convolution2D2 (Convolution2D)   (None, 12, 17, 36)    7812        MaxPool1[0][0]                   
#       ____________________________________________________________________________________________________
#       MaxPool2 (MaxPooling2D)          (None, 6, 8, 36)      0           Convolution2D2[0][0]             
#       ____________________________________________________________________________________________________
#       Convolution2D3 (Convolution2D)   (None, 4, 6, 48)      15600       MaxPool2[0][0]                   
#       ____________________________________________________________________________________________________
#       MaxPool3 (MaxPooling2D)          (None, 2, 3, 48)      0           Convolution2D3[0][0]             
#       ____________________________________________________________________________________________________
#       Dropout (Dropout)                (None, 2, 3, 48)      0           MaxPool3[0][0]                   
#       ____________________________________________________________________________________________________
#       Flatten (Flatten)                (None, 288)           0           Dropout[0][0]                    
#       ____________________________________________________________________________________________________
#       FC2 (Dense)                      (None, 100)           28900       Flatten[0][0]                    
#       ____________________________________________________________________________________________________
#       FC3 (Dense)                      (None, 50)            5050        FC2[0][0]                        
#       ____________________________________________________________________________________________________
#       FC4 (Dense)                      (None, 10)            510         FC3[0][0]                        
#       ____________________________________________________________________________________________________
#       Readout (Dense)                  (None, 1)             0           FC4[0][0]                        
#       ====================================================================================================
#       Total params: 58,544
#       Trainable params: 58,544
#       Non-trainable params: 0
#       ____________________________________________________________________________________________________
#       #+end_example

plot(CarND([160, 320, 3]), to_file="model.png", show_shapes=True)

# Data Pipeline

#       Create a data-processing pipeline.  The 'trainingfile'
#       parameter is the name of a CSV index file specifying samples,
#       with fields for image filenames and for steering angles.  The
#       'base_path' parameter is the directory path for the image
#       filenames.  The pipeline itself is a generator (which is an
#       iterable), where each item from the generator is a batch of
#       samples (X,y).  X and y are each NumPy arrays, with X as a batch
#       of images and y as a batch of outputs.  Finally, augmentation
#       may be performed if a training pipeline is desired, determined
#       by the 'training' parameter.  Training pipelines have their
#       images randomly flipped along the horizontal axis, and are
#       randomly shifted along their horizontal axis.

def pipeline(theta, training=False):
    samples = select(rcycle(fetch(select(split(feed(theta.trainingfile)), [0,3]), theta.base_path)), [0,1])
    if training:
        if theta.flip:
            samples = (rflip(x) for x in samples)
        if theta.shift:
            samples = (rflip(x) for x in samples)
    groups = group(samples, theta.batch_size)
    batches = batch(transpose(groups))
    return batches

# Training

#       Essentially a struct just to gather hyper-parameters into one
#       place, for convenience.

class HyperParameters:
    def __init__(self):
        return

# #+RESULTS:

theta = HyperParameters()
theta.input_shape = [160, 320, 3]
theta.samples_per_epoch = 30
theta.valid_samples_per_epoch = 30
theta.epochs = 3
theta.batch_size = 10
theta.trainingfile = "data/driving_log_overtrain.csv"
theta.validationfile = "data/driving_log_overtrain.csv"
theta.base_path = "data/"
theta.flip = False
theta.shift = False
model = CarND(theta.input_shape)
model.compile(loss="mse", optimizer="adam")
traingen = pipeline(theta, training=True)
validgen = pipeline(theta)
print("")
history = model.fit_generator(
    traingen,
    theta.samples_per_epoch,
    theta.epochs,
    validation_data=validgen,
    verbose=2,
    nb_val_samples=theta.valid_samples_per_epoch)

# #+RESULTS:
#       #+begin_example
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'HyperParameters' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'theta' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'CarND' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'model' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'pipeline' is not defined
#       Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'pipeline' is not defined

#       ... ... ... ... ... Traceback (most recent call last):
# 	File "<stdin>", line 1, in <module>
#       NameError: name 'model' is not defined
# #+end_example

theta = HyperParameters()
theta.input_shape = [160, 320, 3]
theta.trainingfile = "data/driving_log_train.csv"
theta.validationfile = "data/driving_log_validation.csv"
theta.base_path = "data/"
theta.samples_per_epoch = 7000
theta.valid_samples_per_epoch = 7000
theta.epochs = 5
theta.batch_size = 100
theta.flip = False
theta.shift = False
model = CarND(theta.input_shape)
model.compile(loss="mse", optimizer="adam")
print("")
# history = model.fit_generator(
#           traingen,
#           theta.samples_per_epoch,
#           theta.epochs,
#           validation_data=validgen,
#     verbose=2,
#           nb_val_samples=theta.valid_samples_per_epoch)
# model.save_weights("model.h5")
# with open("model.json", "w") as f:
#     f.write(model.to_json())
