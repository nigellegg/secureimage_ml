# -*- coding: utf-8 -*-
# convolutional network to classify gender
# copyright 22015 Chibwe Ltd

"""
This code is an adaptation from the convoluntional network tutorial from
deeplearning.net. It is an simplified version of the "LeNet" approach,
details are described as below:

This implementation simplifies the model in the following ways:
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import os
import sys
import time

import numpy
import cPickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from random import randrange
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn import preprocessing


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def create_shared_dataset(dataset):

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    train_set, test_set = dataset
    test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval


def evaluate_lenet5(datasets, imgh, imgw, nclass, L1_reg=0.00, L2_reg=0.0001,
                    learning_rate=0.01, d=0.0001, n_epochs=300,
                    nkerns=[20, 50], batch_size=500):
    """
    :rtype : object
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = numpy.random.RandomState(23455)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    # x = T.matrix('x')   # the data is presented as rasterized images
    x = T.tensor4('x')
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    # layer0_input = x.reshape((batch_size, 3, 60, 40))
    layer0_input = x.reshape((batch_size, 3, imgh, imgw))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (60-5+1 , 40-5+1) = (56, 36)
    # maxpooling reduces this further to (56/2, 36/2) = (28, 18)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 28, 18)
    #     image_shape=(batch_size, 3, 60, 40),

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, imgh, imgw),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (28-5+1, 18-5+1) = (24, 14)
    # maxpooling reduces this further to (24/2, 14/2) = (12, 7)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 12, 7)
    #     image_shape=(batch_size, nkerns[0], 28, 18),

    lh1 = (imgh - 5 + 1)/2
    lw1 = (imgw-5+1)/2

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], lh1, lw1),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 12 * 7),
    # or (500, 50 * 12 * 7) = (500, 3360) with the default values.
    lh2 = (lh1-5+1)/2
    lw2 = (lw1-5+1)/2

    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * lh2 * lw2,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=nclass)

    ### Regularization
    L1 = (abs(layer0.W).sum()
          + abs(layer1.W).sum()
          + abs(layer2.W).sum()
          + abs(layer3.W).sum())

    L2_sqr = ((layer0.W**2).sum()
              + (layer1.W**2).sum()
              + (layer2.W**2).sum()
              + (layer3.W**2).sum())

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)+L1_reg*L1+L2_reg*L2_sqr

    # create a function to compute the mistakes that are made by the model
    # the following code is modified to suit with the small test set size
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict_model = theano.function(
        [index],
        layer3.y_pred,
        givens={
            x: new_data[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # theano expression to decay the learning rate across epoch
    current_rate = theano.tensor.fscalar('current_rate')

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - current_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index, current_rate],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50  # look at least at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_test_loss = numpy.inf
    learning_rate = numpy.float32(learning_rate)
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False
    test_error = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        learning_rate = learning_rate/(1+d*(epoch-1))
        print "learning rate is %f" % learning_rate

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index, numpy.float32(learning_rate))

            if (iter + 1) % test_frequency == 0:

                # compute zero-one loss on validation set
                test_losses = [test_model(i) for i
                               in xrange(n_test_batches)]
                this_test_loss = numpy.mean(test_losses)

                test_error.append(this_test_loss)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_test_loss * 100.))

                # if we got the best test score until now
                if this_test_loss < best_test_loss:

                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_test_loss = this_test_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_test_loss * 100., best_iter + 1, best_test_loss * 100.))
    print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)

    return params, test_error


def Save_Parameter(model_path, params):
    save_file = open(model_path, 'wb')
    # cPickle.dump(classifier.W.get_value(borrow=True), save_file, -1)
    # cPickle.dump(classifier.b.get_value(borrow=True), save_file, -1)
    cPickle.dump(params, save_file, -1)
    save_file.close()


def get_train_test(data):
    features = data[0]
    labels = data[1]

    #flatten feature to 2d matrix

    # flat_features=features.reshape(384, 60*40*3)

    seed = randrange(100)
    train_x, test_x, train_y, test_y = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=seed)
    return train_x, test_x, train_y, test_y


def load_data(pickle_file):
    load_file = open(pickle_file, 'rb')
    data = cPickle.load(load_file)
    return data


def pickle_data(path, data):
    file = path
    save_file = open(file, 'wb')
    cPickle.dump(data, save_file, -1)
    save_file.close()


def pred_gender():
    folder = os.path.dirname(__file__)
    pickle_file = folder+"/srv/secureimage/pickle_data/image_secure_data.pkl"

    data = load_data(pickle_file)
    img_list = data[0]
    gender_y = data[3]
    age_y = data[4]
    race_y = data[5]

    sss = StratifiedShuffleSplit(gender_y, 1, test_size=0.25, random_state=0)

    for train_index, test_index in sss:
        train_x, test_x = img_list[train_index], img_list[test_index]
        train_y, test_y = gender_y[train_index], gender_y[test_index]
        train_set = [train_x, train_y]
        test_set = [test_x, test_y]
        shuffled_dataset = [train_set, test_set]
        shared_dataset = create_shared_dataset(shuffled_dataset)

        #Gender training
        params, test_error = evaluate_lenet5(shared_dataset, 32, 32, 4)

    test_pickle = xxx
    data = load_data(test_pickle)
    test_data = data[0]

    gend_pred = []
    for i in test_data:
        code = predict_model(i)
        gend_pred.append(code)

    model_file = "/srv/secureimage/model/G_0.2463_54x36_20150210.pkl"
    pickle_data(model_file, params)

    return gend_pred
