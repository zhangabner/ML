import numpy as np
import numpy as numpy
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
batch_size = 64
filter_size1 = 3        
num_filters1 = 4         
filter_size2 = 3        
num_filters2 = 16         
fc_size = 2048
num_channels = 1

def plot_images(images,                  # Images to plot, 2-d array.
                cls_true,                # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):     # Best-net predicted class-no.

    assert len(images) == len(cls_true)
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

if 1:
    tables = pd.io.parsers.read_csv('groundtrueV01.csv',sep=',')
    traintable0 = tables.loc[(tables.indexnum<87531)&(tables.digits==0)& (tables.indexnum!= 68828)]
    x_train_0=np.zeros((6000,img_size_flat))
    y_train_0=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable0.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_0[i]=image
        y_train_0[i]=0

    traintable1 = tables.loc[(tables.indexnum<87531)&(tables.digits==1)]
    x_train_1=np.zeros((6000,img_size_flat))
    y_train_1=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable1.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_1[i]=image
        y_train_1[i]=1

    traintable2 = tables.loc[(tables.indexnum<87531)&(tables.digits==2)]
    x_train_2=np.zeros((6000,img_size_flat))
    y_train_2=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable2.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_2[i]=image
        y_train_2[i]=2

    traintable3 = tables.loc[(tables.indexnum<87531)&(tables.digits==3)]
    x_train_3=np.zeros((6000,img_size_flat))
    y_train_3=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable3.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_3[i]=image
        y_train_3[i]=3

    traintable4 = tables.loc[(tables.indexnum<87531)&(tables.digits==4)]
    x_train_4=np.zeros((6000,img_size_flat))
    y_train_4=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable4.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_4[i]=image
        y_train_4[i]=4

    traintable5 = tables.loc[(tables.indexnum<87531)&(tables.digits==5)]
    x_train_5=np.zeros((6000,img_size_flat))
    y_train_5=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable5.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_5[i]=image
        y_train_5[i]=5

    traintable6 = tables.loc[(tables.indexnum<87531)&(tables.digits==6)]
    x_train_6=np.zeros((6000,img_size_flat))
    y_train_6=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable6.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_6[i]=image
        y_train_6[i]=6

    traintable7 = tables.loc[(tables.indexnum<87531)&(tables.digits==7)]
    x_train_7=np.zeros((6000,img_size_flat))
    y_train_7=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable7.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_7[i]=image
        y_train_7[i]=7

    traintable8 = tables.loc[(tables.indexnum<87531)&(tables.digits==8)]
    x_train_8=np.zeros((6000,img_size_flat))
    y_train_8=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable8.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_8[i]=image
        y_train_8[i]=8
        
    traintable9 = tables.loc[(tables.indexnum<87531)&(tables.digits==9)]
    x_train_9=np.zeros((6000,img_size_flat))
    y_train_9=np.zeros((6000))
    for i in range(0,6000):
        filename = traintable9.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_train_9[i]=image
        y_train_9[i]=9

    testtable0 = tables.loc[(tables.indexnum>=87531)&(tables.digits==0)]
    x_test_0=np.zeros((1000,img_size_flat))
    y_test_0=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable0.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_0[i]=image
        y_test_0[i]=0

    testtable1 = tables.loc[(tables.indexnum>=87531)&(tables.digits==1)]
    x_test_1=np.zeros((1000,img_size_flat))
    y_test_1=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable1.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_1[i]=image
        y_test_1[i]=1

    testtable2 = tables.loc[(tables.indexnum>=87531)&(tables.digits==2)&(tables.indexnum!=96550)&(tables.indexnum!=96527)&(tables.indexnum!=96559)]
    x_test_2=np.zeros((1000,img_size_flat))
    y_test_2=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable2.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_2[i]=image
        y_test_2[i]=2
    cls_predpd = pd.DataFrame(testtable2.indexnum)
    cls_predpd.to_csv("testtable2.csv" ,index=False)

    testtable3 = tables.loc[(tables.indexnum>=87531)&(tables.digits==3)]
    x_test_3=np.zeros((1000,img_size_flat))
    y_test_3=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable3.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_3[i]=image
        y_test_3[i]=3

    testtable4 = tables.loc[(tables.indexnum>=87531)&(tables.digits==4)]
    x_test_4=np.zeros((1000,img_size_flat))
    y_test_4=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable4.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_4[i]=image
        y_test_4[i]=4

    testtable5 = tables.loc[(tables.indexnum>=87531)&(tables.digits==5)]
    x_test_5=np.zeros((1000,img_size_flat))
    y_test_5=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable5.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_5[i]=image
        y_test_5[i]=5

    testtable6 = tables.loc[(tables.indexnum>=87531)&(tables.digits==6)]
    x_test_6=np.zeros((1000,img_size_flat))
    y_test_6=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable6.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_6[i]=image
        y_test_6[i]=6

    testtable7 = tables.loc[(tables.indexnum>=87531)&(tables.digits==7)]
    x_test_7=np.zeros((1000,img_size_flat))
    y_test_7=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable7.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_7[i]=image
        y_test_7[i]=7

    testtable8 = tables.loc[(tables.indexnum>=87531)&(tables.digits==8)]
    x_test_8=np.zeros((1000,img_size_flat))
    y_test_8=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable8.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_8[i]=image
        y_test_8[i]=8
        
    testtable9 = tables.loc[(tables.indexnum>=87531)&(tables.digits==9)]
    x_test_9=np.zeros((1000,img_size_flat))
    y_test_9=np.zeros((1000))
    for i in range(0,1000):
        filename = testtable9.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_test_9[i]=image
        y_test_9[i]=9

#   image = image.reshape(28,28)
#   misc.imshow(image)
    traincombined_x = np.concatenate([x_train_0,x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6,x_train_7,x_train_8,x_train_9], axis=0)
    traincombined_y = np.concatenate([y_train_0,y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6,y_train_7,y_train_8,y_train_9], axis=0)

    idx = np.random.permutation(60000)
    # Select the images and labels for the new training-set.
    x_train = traincombined_x[idx, :]
    y_train = traincombined_y[idx]

    testcombined_x = np.concatenate([x_test_0,x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6,x_test_7,x_test_8,x_test_9], axis=0)
    testcombined_y = np.concatenate([y_test_0,y_test_1,y_test_2,y_test_3,y_test_4,y_test_5,y_test_6,y_test_7,y_test_8,y_test_9], axis=0)
    
    idx = np.random.permutation(10000)
    # Select the images and labels for the new training-set.
    x_test = testcombined_x[idx, :]
    y_test = testcombined_y[idx]
    print (testcombined_y )
    print (y_test )
    cls_predpd = pd.DataFrame(idx)
    cls_predpd.to_csv("idx.csv" ,index=False)
if 1:
    train_X = x_train
    train_y = y_train
    test_X = x_test
    test_y = y_test

    class DataSet(object):
      def __init__(self, images, labels, fake_data=False, one_hot=False,
                   dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
          self._num_examples = 10000
          self.one_hot = one_hot
        else:
          assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape,
                                                     labels.shape))
          self._num_examples = images.shape[0]
    
          # Convert shape from [num examples, rows, columns, depth]
          # to [num examples, rows*columns] (assuming depth == 1)
          assert images.shape[3] == 1
          images = images.reshape(images.shape[0],
                                  images.shape[1] * images.shape[2])
          if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
      @property
      def images(self):
        return self._images
    
      @property
      def labels(self):
        return self._labels
    
      @property
      def num_examples(self):
        return self._num_examples
    
      @property
      def epochs_completed(self):
        return self._epochs_completed
    
      def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
          fake_image = [1] * img_size_flat
          if self.one_hot:
            fake_label = [1] + [0] * 1
          else:
            fake_label = 0
          return [fake_image for _ in xrange(batch_size)], [
              fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
    
    
    def read_data_sets8816(train_dir, fake_data=False, one_hot=False, dtype=tf.uint8):
      class DataSets(object):
        pass
      data_sets = DataSets()

      train_images = train_X.reshape(len(train_X), img_size, img_size, 1)
      train_labels =  np_utils.to_categorical(train_y, num_classes)
      test_images = test_X.reshape(len(test_X), img_size, img_size, 1)
      test_labels = np_utils.to_categorical(test_y, num_classes)
      data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
      data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
      return data_sets
    
    data = read_data_sets8816("data_data/", one_hot=True,dtype=tf.float32)
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    data.train.cls = np.array([label.argmax() for label in data.train.labels])

# Get the first images from the test-set.
images = data.train.images[0:9]

# Get the true classes for those images.
cls_true = data.train.cls[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)

for trainnumbers in range (0,5):
    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))
    
    def new_conv_layer(input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.
    
        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]
    
        # Create new weights aka. filters with the given shape.
        weights = new_weights(shape=shape)
    
        # Create new biases, one for each filter.
        biases = new_biases(length=num_filters)
    
        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
    
        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases
    
        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
    
        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)
    
        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.
    
        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights
    
    def flatten_layer(layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()
    
        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]
    
        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        
        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])
    
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
    
        # Return both the flattened layer and the number of features.
        return layer_flat, num_features
    
    def new_fc_layer(input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?
    
        # Create new weights and biases.
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)
    
        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
    
        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)
    
        return layer
    
    def optimize(num_iterations):
        # Ensure we update the global variable rather than a local copy.
        global total_iterations
    
    
        for i in range(total_iterations,
                       total_iterations + num_iterations):
    
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
    
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
    
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            session.run(optimizer, feed_dict=feed_dict_train)
    
        total_iterations += num_iterations
    
    
    
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True)
    
    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True)
    
    layer_flat, num_features = flatten_layer(layer_conv2)
    
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
    
    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)
    
    y_pred = tf.nn.softmax(layer_fc2)
    
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)
    
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    session = tf.Session()
    
    
    session.run(tf.global_variables_initializer())
    
    train_batch_size = 64
    total_iterations=0
    optimize(num_iterations=50000)  

      
    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}
    
    cls_pred= session.run(y_pred_cls, feed_dict=feed_dict_test)
# write pred to temp files
    filename = 'temp'+str(trainnumbers+1)+'.csv'
    cls_predpd = pd.DataFrame(cls_pred)
    cls_predpd.to_csv(filename ,index=False)    
    session.close()

# analyse temp files    
goodwrinting1 = pd.io.parsers.read_csv('temp1.csv',sep=',')
goodwrinting2 = pd.io.parsers.read_csv('temp2.csv',sep=',')
goodwrinting3 = pd.io.parsers.read_csv('temp3.csv',sep=',')
goodwrinting4 = pd.io.parsers.read_csv('temp4.csv',sep=',')
goodwrinting5 = pd.io.parsers.read_csv('temp5.csv',sep=',')
goodwrinting1.columns = ["cls"]
goodwrinting2.columns = ["cls"]
goodwrinting3.columns = ["cls"]
goodwrinting4.columns = ["cls"]
goodwrinting5.columns = ["cls"]

f4 = open('beforepoisoning.csv', 'w')
tmpline = 'indexnum' +',' +'cls' +',' +'clspredby1' +',' +'clspredby2' +',' +'clspredby3' +',' +'clspredby4' +',' +'clspredby5'   +',' + 'errornum'  '\n' 
f4.write(tmpline)
# count the times of wrong predictions
for tmpi in range (0,len(goodwrinting1)):
    tmpnum = 0
    if  data.test.cls[tmpi] == goodwrinting1.iloc[tmpi].cls:
        tmpnum = tmpnum +1
    if data.test.cls[tmpi] == goodwrinting2.iloc[tmpi].cls:
        tmpnum = tmpnum +1
    if  data.test.cls[tmpi] == goodwrinting3.iloc[tmpi].cls:
        tmpnum = tmpnum +1
    if data.test.cls[tmpi] == goodwrinting4.iloc[tmpi].cls:
        tmpnum = tmpnum +1
    if  data.test.cls[tmpi]== goodwrinting5.iloc[tmpi].cls:
        tmpnum = tmpnum +1
    tmpnum=5-tmpnum
    tmpline =  str(tmpi) +',' +str( data.test.cls[tmpi]) +',' +str(int(goodwrinting1.iloc[tmpi].cls))  +',' +str(int(goodwrinting2.iloc[tmpi].cls))  +',' +str(int(goodwrinting3.iloc[tmpi].cls))  +',' +str(int(goodwrinting4.iloc[tmpi].cls))  +',' +str(int(goodwrinting5.iloc[tmpi].cls))  
    tmpline = tmpline +',' + str(int(tmpnum)) +  '\n' 
    f4.write(tmpline)
f4.close()

tables = pd.io.parsers.read_csv('beforepoisoning.csv',sep=',')
tables2 = tables.loc[(tables.errornum>2)]
for i in range (0,len(tables2)):
    indexnum = tables2.iloc[i].indexnum
    image = data.test.images[indexnum]
    image = image.reshape(28,28)
    misc.imshow(image)
    filename2 =  'check'+  str(i) +  '.png'
    print (filename2)
    misc.imsave(filename2,image)

# there is one in test2  be predicted as 4,  how to locat it and ignore it?    