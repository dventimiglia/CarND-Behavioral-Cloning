# What

# Why

# How

## Approach

## Architecture

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: cannot import name zip_longest
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named keras.layers
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named keras.layers.convolutional
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named keras.models
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named keras.utils.visualize_util
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named scipy.stats
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named cv2
    >>> Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named keras.preprocessing.image
    >>> Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named numpy

### Utilities

Return elements from the iterable.  Shuffle the elements of the
iterable when it becomes exhausted, then begin returning them
again.  Repeat this sequence of operations indefinitely.  Note
that the elements of the iterable are essentially returned in
batches, and that the first batch is not shuffled.  If you want
only to return random elements then you must know batch size,
which will be the number of elements in the underlying finite
iterable, and you must discard the first batch.  The
itertools.islice function can be helpful here.

    def rcycle(iterable):
        saved = []
        for element in iterable:
            yield element
            saved.append(element)
        while saved:
            random.shuffle(saved)
            for element in saved:
                  yield element

Return an iterable over the lines the file with name 'filename'.

    def feed(filename):
        return (l for l in open(filename))

Return an iterable over 'lines', splitting each into records
comprising fields, using 'delimiter'.

    def split(lines, delimiter=","):
        return (line.split(delimiter) for line in lines)

Return an iterable over records of fields, selecting only the
fields listed in 'indices'.

    def select(fields, indices):
        return ([r[i] for i in indices] for r in fields)

Return a NumPy array for the image indicated by 'f'.

    def load(f):
        return np.asarray(Image.open(f))

Return an iterable over 'records', fetching a sample [X,y] for
each record.  A sample is a ordered pair with the first element
X a NumPy array (typically, an image) and the second element y
floating point number (the label).

    def fetch(records, base):
        return ([load(base+f.strip()) for f in record[:1]]+[float(v) for v in record[1:]] for record in records)

Randomly flip an image 'x' along its horizontal axis 50% of the
time.

    def rflip(x):
        return x if random.choice([True, False]) else [img.flip_axis(x[0],1), -1*x[1]]

Randomly shift an image 'x' along its horizontal axis by
'factor' amount.

    def rshift(x, factor=0.1):
        return img.random_shift(x, factor, 0.0, 0, 1, 2, fill_mode='wrap')

Iterate over 'items' but return them in groups of size 'n'.  If
need be, fill the last group with 'fillvalue'.

    def group(items, n, fillvalue=None):
        return zip_longest(*([iter(items)]*n), fillvalue=fillvalue)

Transpose items in the 'tuples' iterable.  Each item is expected
to be a tuple and all tuples are expected to have the same
length.  The transposition is such that for each position 'i'
within the tuples, all of the elements at that position across
all the items are themselves grouped together.  Each group is
realized into a list.  If 'tuples' contains m items and each
item is itself a tuple of n elements, then what is returned is a
set of n lists, and each list contains m elements.  The n lists
are themselves presented as an iterable.

    def transpose(tuples):
        return (list(map(list, zip(*g))) for g in tuples)

Iterate over 'groups', each of which is itself an iterable (such
as a list), and turn the groups selected by 'indices' into a
NumPy array.  Naturally, the groups are expected to be of items
that are compatible with NumPy arrays, which would be any of the
appropriate numeric types.

    def batch(groups, indices=[0, 1]):
        return ([np.asarray(t[i]) for i in indices] for t in groups)

### Model

Return a Keras neural network model.

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

    CarND([160, 320, 3]).summary()

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 2, in CarND
    NameError: global name 'Sequential' is not defined

    plot(CarND([160, 320, 3]), to_file="model.png", show_shapes=True)

![img](model.png "CarND Neural-Net Architecture")

## Data

### Collection and Preparation

    wget -nc "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
    unzip data.zip > /dev/null 2>&1
    rm -rf __MACOSX

    cat data/driving_log.csv | tail -n+2 | shuf > data/driving_log_all.csv
    cat data/driving_log_all.csv | head -n7000 > data/driving_log_train.csv
    cat data/driving_log_all.csv | tail -n+7000 > data/driving_log_validation.csv

    wc -l data/driving_log.csv
    wc -l data/driving_log_train.csv
    wc -l data/driving_log_validation.csv

    8037 data/driving_log.csv
    7000 data/driving_log_train.csv
    1037 data/driving_log_validation.csv

### Characteristics

### Examples

## Training

### Data Pipeline

Create a data-processing pipeline.  The 'training<sub>index'</sub>
parameter is the name of a CSV index file specifying samples,
with fields for image filenames and for steering angles.  The
'base<sub>path'</sub> parameter is the directory path for the image
filenames.  The pipeline itself is a generator (which is an
iterable), where each item from the generator is a batch of
samples (X,y).  X and y are each NumPy arrays, with X as a batch
of images and y as a batch of outputs.  Finally, augmentation
may be performed if a training pipeline is desired, determined
by the 'training' parameter.  Training pipelines have their
images randomly flipped along the horizontal axis, and are
randomly shifted along their horizontal axis.

    def pipeline(theta, training=False):
        samples = select(rcycle(fetch(select(split(feed(theta.training_index)), [0,3]), theta.base_path)), [0,1])
        if training:
            if theta.flip:
                samples = (rflip(x) for x in samples)
            if theta.shift:
                samples = ((rshift(x[0]),x[1]) for x in samples)
        groups = group(samples, theta.batch_size)
        batches = batch(transpose(groups))
        return batches

### Training

Train the model.

    def train(model):
        traingen = pipeline(theta, training=True)
        validgen = pipeline(theta)
        history = model.fit_generator(
            traingen,
            theta.samples_per_epoch,
            theta.epochs,
            validation_data=validgen,
            verbose=2,
            nb_val_samples=theta.valid_samples_per_epoch)

### Data Structures

Essentially a struct just to gather hyper-parameters into one
place, for convenience.

    class HyperParameters:
        def __init__(self):
            return

### Entry-point

    if __name__=="__main__":        # In case this module is imported
        theta = HyperParameters()
        theta.input_shape = [160, 320, 3]
        theta.samples_per_epoch = 300
        theta.valid_samples_per_epoch = 300
        theta.epochs = 3
        theta.batch_size = 100
        theta.training_index = "data/driving_log_overtrain.csv"
        theta.validation_index = "data/driving_log_overtrain.csv"
        theta.base_path = "data/"
        theta.flip = False
        theta.shift = False
        if sys.argv[0]!='':         # Running from the command line
            theta.training_index = os.environ['TRAINING_INDEX']
            theta.validation_index = os.environ['VALIDATION_INDEX']
            theta.base_path = os.environ['BASE_PATH']
            theta.samples_per_epoch = int(os.environ['SAMPLES_PER_EPOCH'])
            theta.valid_samples_per_epoch = int(os.environ['VALID_SAMPLES_PER_EPOCH'])
            theta.epochs = int(os.environ['EPOCHS'])
            theta.batch_size = int(os.environ['BATCH_SIZE'])
            theta.flip = os.environ['FLIP']=='yes'
            theta.shift = os.environ['SHIFT']=='yes'
        model = CarND(theta.input_shape)
        model.compile(loss="mse", optimizer="adam")
        print("")
        train(model)
        model.save_weights("model.h5")
        with open("model.json", "w") as f:
            f.write(model.to_json())
        gc.collect()

    ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... Traceback (most recent call last):
      File "<stdin>", line 23, in <module>
      File "<stdin>", line 2, in CarND
    NameError: global name 'Sequential' is not defined
