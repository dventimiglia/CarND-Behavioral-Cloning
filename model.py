
# #+RESULTS:

#       Having set up the Conda environment, and activated it, now we
#       can finally load the Python modules that we will need in later
#       sections.

from PIL import Image
from itertools import groupby, islice, zip_longest, cycle, filterfalse
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Lambda, AveragePooling2D
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.models import Sequential, model_from_json
from keras.utils.visualize_util import plot
from scipy.stats import kurtosis, skew, describe
import gc
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import random

# Utilities

#       Another piece of advice impressed upon students in the class, to
#       the point of it practically being a requirement, was to learn to
#       use Python [[https://wiki.python.org/moin/Generators][generators]] and the [[https://keras.io/models/sequential/][=fit_generator=]] function in our
#       Deep Learning toolkit, [[https://keras.io/][Keras]].  Generators allow for a form of
#       [[https://en.wikipedia.org/wiki/Lazy_loading][lazy loading]], which can be useful in Machine Learning settings
#       where large data sets that do not fit into main memory are the
#       norm.  Keras makes use of that with =fit_generator=, which
#       expects input presented as generators that infinitely recycle
#       over the underlying data.

#       I took that advice to heart and spent considerable
#       time---perhaps more than was necessary---learning about
#       generators and generator expressions.  It did pay off somewhat
#       in that I developed a tiny library of reusable, composeable
#       generator expressions, which are presented here.

#       Before doing that, though, first a detour and a bit of advice.
#       Anyone who is working with Python generators is urged to become
#       acquainted with [[https://docs.python.org/3/library/itertools.html][=itertools=]], a standard Python library of
#       reusable, composeable generators.  For me the [[https://docs.python.org/3/library/itertools.html#itertools.cycle][=cycle=]] generator
#       was a key find.  As mentioned above, =fit_generator= needs
#       infinitely-recycling generators, which is exactly what
#       =itertools.cycle= provides.  One wrinkle is that =cycle=
#       accomplishes this with an internal data cache, so if your data
#       do /not/ fit in memory you may have to seek an alternative.
#       However, if your data /do/ fit in memory this confers a very
#       nice property, for free: after cycling through the data the
#       first time all subsequent retrievals are from memory, so that
#       performance improves dramatically after the first cycle.

#       This turns out to be very beneficial and entirely appropriate
#       for our problem.  Suppose we use the Udacity data.  In that
#       case, we have 8136 images (training + validation) provided we
#       use one camera only (such as the center camera).  As we shall
#       see below, each image is a 320x160 pixel array of RGB values,
#       for a total of 150k per image.  That means approximately 1 GB of
#       RAM is required to store the Udacity data.  My 4 year-old laptop
#       has 4 times that.  Now, this rosy picture might quickly dissolve
#       if we use much more data, such as by using the other camera
#       angles and/or acquiring more training data.  Then again, it may
#       not.  With virtual memory, excess pages /should/ be swapped out
#       to disk.  That's not ideal, but to first order it's not obvious
#       that it's functionally much different from or much worse than
#       recycling the data by repeatedly loading the raw image files.
#       In fact, it may be better, since at least we only perform the
#       actual translation from PNG format to [[http://www.numpy.org/][NumPy]] data arrays once for
#       each image.

#       If we are really concerned about memory usage we might consider
#       reducing the input image size, such as with OpenCV's
#       [[http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize][=cv2.resize=]] command, and we might consider cropping the image.
#       But, I think we should think carefully about this.  These
#       operations may have an effect on the performance of the model,
#       and so manipulating the images in this way is not something we
#       should take lightly.  Nevertheless, it can be beneficial as we
#       shall see shortly.  However, if we /do/ decide to crop and
#       resize, there is a technical trade-off to be made.  Either we
#       can crop and resize as a pre-processing step, or we can do it
#       directly within the model, and there are advantages and
#       disadvantages to each.  If we crop and resize as a
#       pre-processing step, it has direct impact on the aforementioned
#       memory considerations.  But, we /must take care to perform exactly the same crop and resize operations in the network server!/  Since cropping and resizing essentially introduce new
#       hyper-parameters, those parameters somehow must be communicated
#       to =drive.py=.  If we crop and resize directly within the model,
#       it has no beneficial impact on the aforementioned memory
#       considerations.  But, /we get those operations and their internal hyper-parameters for free within the network server!/  I found
#       the latter advantage to be much greater and so the trade-off I
#       selected was to crop and resize within the model.

#       In any case, my experiments showed that for the Udacity data at
#       least, loading the data into an in-memory cache via
#       =itertools.cycle= (or, more precisely, my variation of it) and
#       then infinitely recycling over them proved to be a very good
#       solution.

#       However, there is one problem with =itertools.cycle= by itself,
#       and that is that again, according to Deep Learning lore, it is
#       prudent to randomize the data on every epoch.  To do that, we
#       need to rewrite =itertools.cycle= so that it shuffles the data
#       upon every recycle.  That is easily done, as shown below.  Note
#       that the elements of the iterable are essentially returned in
#       batches, and that the first batch is not shuffled.  If you want
#       only to return random elements then you must know the batch
#       size, which will be the number of elements in the underlying
#       finite iterable, and you must discard the first batch.  The
#       itertools.islice function can be helpful here.  In our case it
#       is not a problem since all of the data were already shuffled
#       once using Unix command-line utilities above.

def rcycle(iterable):
    saved = []                 # In-memory cache
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)  # Shuffle every batch
        for element in saved:
              yield element

# #+RESULTS:

#       If we invoke =rcycle= on a sequence drawn from the interval
#       [0,5), taken in 3 batches for a total of 15 values we can see
#       this behavior.  The first 5 values are drawn in order, but then
#       the next 10 are drawn in two batches, each batch shuffled
#       independently.  In practice, this is not a problem.

[x for x in islice(rcycle(range(5)), 15)]

# #+RESULTS:
#       : [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 3, 0, 2, 1, 4]

#       The remaining utility functions that I wrote are quite
#       straightforward and for brevity are written as "one-liners."

#       - feed :: generator that feeds lines from the file named by 'filename'
#       - split :: generator that splits lines into tuples based on a delimiter
#       - select :: generator that selects out elements from tuples
#       - load :: non-generator that reads an image file into a NumPy array
#       - Xflip :: non-generator that flips an input image horizontally
#       - yflip :: non-generator that flips a target label to its negative
#       - rmap :: generator that randomly applies or does not apply a
#                 function with equal probability
#       - rflip :: generator that flips samples 50% of the time
#       - fetch :: generator that loads index file entries into samples
#       - group :: generator that groups input elements into lists
#       - transpose :: generator that takes a generator of lists into a
#                      list of generators
#       - batch :: generator that takes a list of generators into a list
#                  of NumPy array "batches"

#       The actual code for these utility functions is presented below.

feed = lambda filename: (l for l in open(filename))
split = lambda lines, delimiter=",": (line.split(delimiter) for line in lines)
select = lambda fields, indices: ([r[i] for i in indices] for r in fields)
load = lambda f: np.asarray(Image.open(f))
Xflip = lambda x: x[:,::-1,:]
yflip = lambda x: -x
sflip = lambda s: (Xflip(s[0]), yflip(s[1]))
rmap = lambda f,g: (x if random.choice([True, False]) else f(x) for x in g)
rflip = lambda s: rmap(sflip, s)
fetch = lambda records, base: ([load(base+f.strip()) for f in record[:1]]+[float(v) for v in record[1:]] for record in records)
group = lambda items, n, fillvalue=None: zip_longest(*([iter(items)]*n), fillvalue=fillvalue)
transpose = lambda tuples: (list(map(list, zip(*g))) for g in tuples)
batch = lambda groups, indices=[0, 1]: ([np.asarray(t[i]) for i in indices] for t in groups)

# Exploratory Analysis

#       It often pays to explore your data with relatively few
#       constraints before diving in to build and train the actual
#       model.  One may gain insights that help guide you to better
#       models and strategies, and avoid pitfalls and dead-ends.  

#       To that end, first we just want to see what kind of input data
#       we are dealing with.  We know that they are RGB images, so let's
#       load a few of them for display.  Here, we show the three frames
#       taken from the =driving_log_overtrain.csv= file described
#       above---center camera only---labeled by their corresponding
#       steering angles.  As you can see, the image with a large
#       negative angle seems to have the car on the extreme right edge
#       of the road.  Perhaps the driver in this situation was executing
#       a "recovery" maneuver, turning sharply to the left to veer away
#       from the road's right edge and back to the centerline.
#       Likewise, with the next figure that has a large positive angle,
#       we see that the car appears to be on the extreme left edge of
#       the road.  Perhaps the opposite recovery maneuver was in play.
#       Finally, in the third and last image that has a neutral steering
#       angle (0.0), the car appears to be sailing right down the middle
#       of the road, a circumstance that absent extraneous circumstances
#       (other cars, people, rodents) should not require corrective
#       steering.

X = [x for x in fetch(select(split(feed("data/driving_log_overtrain.csv")), [0,3]), "data/")]
f = plt.figure()                        # start a figure
plt.imshow(X[0][0])                     # show the image
f.suptitle("Angle: " + str(X[0][1]))    # add figure title
s = plt.savefig("road1.png", format="png", bbox_inches='tight')
f = plt.figure()
plt.imshow(X[1][0])
f.suptitle("Angle: " + str(X[1][1]))
s = plt.savefig("road2.png", format="png", bbox_inches='tight')
f = plt.figure()
plt.imshow(X[2][0])
f.suptitle("Angle: " + str(X[2][1]))
s = plt.savefig("road3.png", format="png", bbox_inches='tight')

# #+RESULTS:
#       : Text(0.5,0.98,'Angle: 0.0')

#       #+CAPTION: Large Negative Steering Angle
#       [[file:road1.png]]

#       #+CAPTION: Large Positive Steering Angle
#       [[file:road2.png]]

#       #+CAPTION: Neutral Steering Angle
#       [[file:road3.png]]

#       Next, we get the shape of an image which, as we said above, is
#       320x160x3.  In NumPy parlance that's =(160, 320, 3)=, for 160
#       rows (the y direction), 320 columns (the x direction), and 3
#       channels (the RGB colorspace).

print(X[0][0].shape)  # X[0] is an (image,angle) sample. X[0][0] is just the image

# #+RESULTS:
#       : (160, 320, 3)

#       We can see that the images naturally divide roughly into "road"
#       below the horizon and "sky" above the horizon, with background
#       scenery (trees, mountains, etc.) superimposed onto the sky.
#       While the sky (really, the scenery) might contain useful
#       navigational information, it is plausible that it contains
#       little or no useful information for the simpler task of
#       maintaining an autonomous vehicle near the centerline of a
#       track, a subject we shall return to later.  Likewise, it is
#       almost certain that the small amount of car "hood" superimposed
#       onto the bottom of the images contains no useful information.
#       Therefore, let us see what the images would look like with the
#       hood cropped out on the bottom by 20 pixels, and the sky cropped
#       out on the top by [[(sky60)][60 pixels]], [[(sky80)][80 pixels]], and [[(sky100)][100 pixels]].

f = plt.figure()
plt.imshow(X[0][0][60:140])    # sky:60 
s = plt.savefig("road4.png", format="png", bbox_inches='tight')
plt.imshow(X[0][0][80:140])    # sky:80 
s = plt.savefig("road5.png", format="png", bbox_inches='tight')
plt.imshow(X[0][0][100:140])   # sky:100 
s = plt.savefig("road6.png", format="png", bbox_inches='tight')

# #+RESULTS:
#       : 
#       : <matplotlib.image.AxesImage object at 0x7f691b70dcf8>
#       : <matplotlib.image.AxesImage object at 0x7f691b70de80>
#       : <matplotlib.image.AxesImage object at 0x7f691b757ac8>

#       #+CAPTION: Hood Crop: 20, Sky Crop:  60
#       [[file:road4.png]]

#       #+CAPTION: Hood Crop: 20, Sky Crop:  80
#       [[file:road5.png]]

#       #+CAPTION: Hood Crop: 20, Sky Crop:  100
#       [[file:road6.png]]

#       I should pause here to address the issue of why we are only
#       using the center camera.  After all, the training data do
#       provide two additional camera images: the left and right
#       cameras.  Surely, those provide additional useful information
#       that the model potentially could make use of.  However, in my
#       opinion there is a serious problem in using these data: the
#       simulator seems only to send to the network server in
#       =drive.py= the center image.  I really do not understand why
#       this is the case, since obviously the simulator is fully-capable
#       of scribbling the extra camera outputs down when recording
#       training data.  

#       Now, I have observed considerable discussion on the Slack
#       channel for this course on the subject of using the additional
#       camera images (left and right) along with a corresponding shift
#       of the steering angles.  Frankly, I am skeptical about the merit
#       of this strategy and shall avoid this practice, evaluate the
#       outcome, and adopt it only if absolutely necessary.

#       We begin by conducting a very simple analysis of the target
#       labels, which again are steering angles in the interval [-1,
#       1].  In fact, as real-valued outputs it may be a stretch to call
#       them "labels" and this is not really a classification problem.
#       Nevertheless in the interest of time we will adopt the term.

f = plt.figure()                   
y1 = np.array([float(s[0]) for s in select(split(feed("data/driving_log_all.csv")),[3])])
h = plt.hist(y1,bins=100)          # plot histogram
s = plt.savefig("hist1.png", format='png', bbox_inches='tight')
print("")
pp.pprint(describe(y1)._asdict())  # print descriptive statistics

# #+RESULTS:
#       : 
#       : >>>
#       : OrderedDict([('nobs', 8036),
#       :              ('minmax', (-0.94269539999999996, 1.0)),
#       :              ('mean', 0.0040696440648332506),
#       :              ('variance', 0.016599764281272529),
#       :              ('skewness', -0.13028924577521916),
#       :              ('kurtosis', 6.311554102057668)])

#       #+CAPTION: All Samples - No Reflection
#       [[file:hist1.png]]

#       The data have non-zero /mean/ and /skewness/, perhaps arising
#       from a bias toward left-hand turns when driving on a closed
#       track.

#       The data are dominated by small steering angles because the car
#       spends most of its time on the track in straightaways.  The
#       asymmetry in the data is more apparent if I mask out small
#       angles and repeat the analysis.  Steering angles occupy the
#       interval [-1, 1], but the "straight" samples appear to be within
#       the neighborhood [-0.01, 0.01].

#       We might consider masking out small angled samples from the
#       actual training data as well, a subject we shall return to in
#       the summary.

f = plt.figure()
p = lambda x: abs(x)<0.01
y2 = np.array([s for s in filterfalse(p,y1)])
h = plt.hist(y2,bins=100)
s = plt.savefig("hist2.png", format='png', bbox_inches='tight')
print("")
pp.pprint(describe(y2)._asdict())

# #+RESULTS:
#       : 
#       : >>> >>> >>>
#       : OrderedDict([('nobs', 3584),
#       :              ('minmax', (-0.94269539999999996, 1.0)),
#       :              ('mean', 0.0091718659514508916),
#       :              ('variance', 0.037178302717086116),
#       :              ('skewness', -0.1665782596901517),
#       :              ('kurtosis', 1.1768785967587378)])

#       #+CAPTION: abs(angle)>0.01 - No Reflection
#       [[file:hist2.png]]

#       A simple trick we can play to remove this asymmetry---if we
#       wish---is to join the data with its reflection, effectively
#       doubling our sample size in the process.  For illustration
#       purposes only, we shall again mask out small angle samples.

f = plt.figure()
y3 = np.append(y2, -y2)
h = plt.hist(y3,bins=100)
s = plt.savefig("hist3.png", format='png', bbox_inches='tight')
print("")
pp.pprint(describe(y3)._asdict())

# #+RESULTS:
#       : 
#       : >>> >>>
#       : OrderedDict([('nobs', 7168),
#       :              ('minmax', (-1.0, 1.0)),
#       :              ('mean', 0.0),
#       :              ('variance', 0.03725725015081123),
#       :              ('skewness', 0.0),
#       :              ('kurtosis', 1.1400026599654973)])

#       #+CAPTION: abs(angle)>0.01 - Full Reflection
#       [[file:hist3.png]]

#       In one of the least-surprising outcomes of the year, after
#       performing the reflection and joining operations, the data now
#       are symmetrical, with mean and skewness identically 0.

#       Of course, in this analysis I have only reflected the target
#       labels.  If I apply this strategy to the training data,
#       naturally I need to reflect along their horizontal axes the
#       corresponding input images as well.  In fact, that is the
#       purpose of the =Xflip=, =yflip=, =rmap=, =rflip=, and =sflip=
#       utility functions described above.

#       It turns out there is another approach to dealing with the bias
#       and asymmetry in the training data.  In lieu of reflecting the
#       data, which by definition imposes a 0 mean and 0 skewness, we
#       can instead just randomly flip samples 50% of the time.  While
#       that will not yield a perfectly balanced and symmetric data
#       distribution, given enough samples it should give us a crude
#       approximation.  Moreover, it saves us from having to store more
#       images in memory, at the cost of some extra computation.
#       Essentially, we are making the classic space-time trade-off
#       between memory consumption and CPU usage.

y4 = [y for y in rmap(yflip, islice(cycle(y2), 2*y1.shape[0]))]  # 2 batches
y5 = [y for y in rmap(yflip, islice(cycle(y2), 4*y1.shape[0]))]  # 4 batches
y6 = [y for y in rmap(yflip, islice(cycle(y2), 8*y1.shape[0]))]  # 8 batches
f = plt.figure()
h = plt.hist(y4,bins=100)
s = plt.savefig("hist5.png", format='png', bbox_inches='tight')
f = plt.figure()
h = plt.hist(y5,bins=100)
s = plt.savefig("hist6.png", format='png', bbox_inches='tight')
f = plt.figure()
h = plt.hist(y6,bins=100)
s = plt.savefig("hist7.png", format='png', bbox_inches='tight')
print("")
pp.pprint(describe(y4)._asdict())
print("")
pp.pprint(describe(y5)._asdict())
print("")
pp.pprint(describe(y6)._asdict())

# #+RESULTS:
#       #+begin_example

#       >>>
#       OrderedDict([('nobs', 16072),
# 		   ('minmax', (-1.0, 1.0)),
# 		   ('mean', -0.0012464327762568444),
# 		   ('variance', 0.037297357483824768),
# 		   ('skewness', 0.03935649430167018),
# 		   ('kurtosis', 1.10078365162724)])

#       OrderedDict([('nobs', 32144),
# 		   ('minmax', (-1.0, 1.0)),
# 		   ('mean', 0.0013097329084743655),
# 		   ('variance', 0.037230073594916549),
# 		   ('skewness', -0.011391695445489564),
# 		   ('kurtosis', 1.1405030969598942)])

#       OrderedDict([('nobs', 64288),
# 		   ('minmax', (-1.0, 1.0)),
# 		   ('mean', -9.7916912643106218e-05),
# 		   ('variance', 0.03724814334901478),
# 		   ('skewness', -0.004653875002642311),
# 		   ('kurtosis', 1.1415231296661128)])
# #+end_example

#       Here, we see that as we increase the number of samples we draw
#       from the underlying data set, while randomly flipping them, the
#       mean tends to diminish.  The skewness does not behave quite so
#       well, though a coarser smoothing kernel (larger bin sizes for
#       the histograms) may help.  In any case, the following figures do
#       suggest that randomly flipping the data and drawing larger
#       sample sizes does help balance out the data.

#       #+CAPTION: abs(angle)>0.01 - Random Flipping, Recycle: 2
#       [[file:hist5.png]]

#       #+CAPTION: abs(angle)>0.01 - Random Flipping, Recycle: 4
#       [[file:hist6.png]]

#       #+CAPTION: abs(angle)>0.01 - Random Flipping, Recycle: 8
#       [[file:hist7.png]]

#       The =sflip= utility function defined above flips not only the
#       target labels---the steering angles---but also the images (as it
#       must).  We check that by again displaying the 3 samples from
#       =driving_log_overtrain.csv= as above, but this time with each of
#       them flipped.

X2 = [sflip(x) for x in X]
f = plt.figure()                        
plt.imshow(X2[0][0])                     
f.suptitle("Angle: " + str(X2[0][1]))    
s = plt.savefig("road7.png", format="png", bbox_inches='tight')
f = plt.figure()
plt.imshow(X2[1][0])
f.suptitle("Angle: " + str(X2[1][1]))
s = plt.savefig("road8.png", format="png", bbox_inches='tight')
f = plt.figure()
plt.imshow(X2[2][0])
f.suptitle("Angle: " + str(X2[2][1]))
s = plt.savefig("road9.png", format="png", bbox_inches='tight')

# Model

#       The actual model is laid out in Keras code below.  It is coded
#       as a function that returns a Keras =model=.  Note that the
#       function does take the =input_shape= and the =crop_shape=.
#       Though I could perform image resizing (such as with
#       =cv2.resize=) and cropping outside of the model, I actually do
#       them inside the model.  This has several advantages.

#       1. It simplifies the code.
#       2. Whatever cropping/resizing occurs /must/ also be done in the
#          network service.  Performing these operations with the model
#          means that this is handled automatically, for free.
#       3. In general, we might realize better training performance as
#          the cropping/resizing occur in the GPUs rather than in the
#          CPU.  In my particular case this did not occur because I
#          trained only on a laptop without a GPU.

#       This means that the =input_shape= is not really a
#       hyper-parameter, since it is just the original image size, which
#       as we already have seen is (160, 320, 3).  The =crop_shape=
#       still is a hyper-parameter, of course.

def CarND(input_shape, crop_shape):
    model = Sequential()
 
    # Crop
    # model.add(Cropping2D(((80,20),(1,1)), input_shape=input_shape, name="Crop"))
    model.add(Cropping2D(crop_shape, input_shape=input_shape, name="Crop"))
 
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

#       Here is a summary of the actual model, as generated directly by
#       =model.summary= in Keras.

CarND([160, 320, 3], ((80,20),(1,1))).summary()

# #+RESULTS:
#       #+begin_example
#       ____________________________________________________________________________________________________
#       Layer (type)                     Output Shape          Param #     Connected to                     
#       ====================================================================================================
#       Crop (Cropping2D)                (None, 60, 318, 3)    0           cropping2d_input_1[0][0]         
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
#       Readout (Dense)                  (None, 1)             11          FC4[0][0]                        
#       ====================================================================================================
#       Total params: 58,555
#       Trainable params: 58,544
#       Non-trainable params: 11
#       ____________________________________________________________________________________________________
# #+end_example

#       And, here is a visualization of the model, as provided by the
#       =plot= function in Keras.

plot(CarND([160, 320, 3], ((80,20),(1,1))), to_file="model.png", show_shapes=True)

# Data Pipeline

#       The data-processing pipeline is rather simple, given the
#       composeable generator and non-generator utility functions
#       defined above.  The only real wrinkle is that we may want to do
#       random sample flipping (of both the image and its corresponding
#       steering angle) during training but we probably do not need to
#       do so for the validation data.  Since assembling the pipeline is
#       otherwise very similar for both the training and validation
#       data, and both should result in a generator that yields batches
#       consumable by the [[https://keras.io/models/sequential/][=model.fit_generator=]] function of Keras's
#       Sequential model, both pipelines are assembled in a function.

def pipeline(theta, training=False):
    # randomly cycle through cached, loaded samples (images + angles)
    samples = select(rcycle(fetch(select(split(feed(theta.trainingfile)), [0,3]), theta.base_path)), [0,1])
    # for training we might do sample flipping but no need for validation
    if training:
        if theta.flip:
            samples = (sflip(x) for x in samples)
    # group the samples
    groups = group(samples, theta.batch_size)
    # turn the groups into batches (NumPy arrays)
    batches = batch(transpose(groups))
    # return the batch generator
    return batches

# Training

#       With functions for constructing the model architecture and the
#       data pipelines in hand, training is very simple.  One additional
#       item to point out is this.  There can be a proliferation of
#       /literal/ hyper-parameters (crop sizes, epochs, batch sizes,
#       whether or not to do random flipping, etc.) and passing these
#       parameters among all the related and nested functions can be a
#       nuisance.  I find it convenient to collect such parameters into
#       a handy data structure that can be instantiated as a global
#       variable, and then all of the functions can access the
#       parameters that they need.  In Python, two obvious candidates
#       that leap to mind are [[https://docs.python.org/3/tutorial/datastructures.html][dictionaries]] and [[https://docs.python.org/3/tutorial/classes.html][classes]].  Python
#       dictionaries are slightly easier to create, but slightly more of
#       a nuisance to use, whereas the reverse is true for Python
#       classes/objects.  I elected to use an global instance =theta= of
#       a class =HyperParameters=, but this is not considered to be a
#       very important point.

#       Another point worth discussing relates to "cropping."  As
#       discussed above in the data analysis and architecture sections
#       and below in the summary, cropping of the data is an option that
#       I took.  The [[(crop_shape)][=theta.crop_shape=]] hyper-parameter that appears
#       below sets the cropping window, and it works as follows.

#       The =theta.crop_shape= parameter is dictated by the [[https://keras.io/layers/convolutional/#cropping2d][Cropping2D]]
#       layer to be a tuple of tuples.  The first tuple sets the number
#       of pixels to be cropped from the top and bottom edges along the
#       image height direction, and the second tuple sets the number of
#       pixels to be cropped from the left and right edges of the width
#       direction.  As you can see, the top and bottom are cropped with
#       =(80, 20)= which as in the figure above, removes much of the
#       image above the horizon and some of the image where the car hood
#       is superimposed.  Those values were chosen by experimentation,
#       and the choices are discussed below.  However, the =(1,1)= value
#       for the width cropping merits some explanation.  In principle, I
#       saw no reason to crop anything in the width direction.  However,
#       Keras has a bug (fixed but not yet in a stable release) such
#       that Cropping2D fails when any of the four elements in the tuple
#       of tuples is 0.  This is the reason for the 1 pixel crops on the
#       left and right edges.

#       Now, as a sanity check, we conduct a small training (3 epochs,
#       30 samples per epoch, batch size 10) of the data in
#       =driving_log_overtrain.csv=.  This is just to "get our feet wet"
#       and quickly to verify that the code written above even works.
#       Note that we use the same file for the validation set.  This is
#       just a test, so it does not really matter what we use for the
#       validation set.

class HyperParameters:
    def __init__(self):
        return

theta = HyperParameters()
theta.input_shape = [160, 320, 3]
theta.crop_shape = ((80,20),(1,1))   # crop size 
theta.samples_per_epoch = 30
theta.valid_samples_per_epoch = 30
theta.epochs = 3
theta.batch_size = 10
theta.trainingfile = "data/driving_log_overtrain.csv"
theta.validationfile = "data/driving_log_overtrain.csv"
theta.base_path = "data/"
theta.flip = False

model = CarND(theta.input_shape, theta.crop_shape)
model.compile(loss="mse", optimizer="adam")

traingen = pipeline(theta, training=True)
validgen = pipeline(theta)

print("")
history = model.fit_generator(
    traingen,
    theta.samples_per_epoch,
    theta.epochs,
    validation_data=validgen,
    verbose=1,
    nb_val_samples=theta.valid_samples_per_epoch)

# #+RESULTS:
#       : 
#       : ... ... >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
#       : ... ... ... ... ... ... Epoch 1/3
#       : 10/30 [=========>....................] - ETA: 3s - loss: 0.626220/30 [===================>..........] - ETA: 0s - loss: 0.562630/30 [==============================] - 1s - loss: 0.5480 - val_loss: 0.4095
#       : Epoch 2/3
#       : 10/30 [=========>....................] - ETA: 0s - loss: 0.424120/30 [===================>..........] - ETA: 0s - loss: 0.380430/30 [==============================] - 0s - loss: 0.3636 - val_loss: 0.2617
#       : Epoch 3/3
#       : 10/30 [=========>....................] - ETA: 0s - loss: 0.288220/30 [===================>..........] - ETA: 0s - loss: 0.247530/30 [==============================] - 0s - loss: 0.2292 - val_loss: 0.1897

#       Next, we perform the actual training on the
#       =driving_log_train.csv= file, validating against the
#       =driving_log_validation.csv= file.  After /this/ training we
#       actually save the model to =model.json= and the model weights to
#       =model.h5=, files suitable for input into the network service in
#       =drive.py=.

theta = HyperParameters()
theta.input_shape = [160, 320, 3]
theta.crop_shape = ((80,20),(1,1))
theta.trainingfile = "data/driving_log_train.csv"
theta.validationfile = "data/driving_log_validation.csv"
theta.base_path = "data/"
theta.samples_per_epoch = 14000
theta.valid_samples_per_epoch = 1036
theta.epochs = 3
theta.batch_size = 100
theta.flip = True

model = CarND(theta.input_shape, theta.crop_shape)
model.compile(loss="mse", optimizer="adam")

traingen = pipeline(theta, training=True)
validgen = pipeline(theta)

print("")
history = model.fit_generator(
    traingen,
    theta.samples_per_epoch,
    theta.epochs,
    validation_data=validgen,
    verbose=1,
    nb_val_samples=theta.valid_samples_per_epoch)
model.save_weights("model.h5")
with open("model.json", "w") as f:
    f.write(model.to_json())
# gc.collect()
