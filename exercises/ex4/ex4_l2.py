import sys
from os import path
import timeit

import theano
import theano.tensor as T
from theano.printing import pydotprint
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import pylab as pl

sys.path.append("../utils")
from data_loader import DataLoader
from mlp import MLP

def do_gd(etaVal, epochs, layers, train_set, 
        valid_set=None, test_set=None, L2_reg=0, batch_size=100, scale=1):
    '''
    batch_size = 100
    0 L2 regularization (by default)
    function returns training error and validation error after each epoch
    '''
    SEED = 5318
    np.random.seed(SEED)
    X = T.matrix('X')
    Y = T.ivector('Y')
    index = T.lscalar('index')
    eta = T.fscalar('eta')
    
    n_in = layers[0]
    n_out = layers[-1]

    # Get the datasets
    trainX, trainY = train_set
    validX, validY = valid_set
    testX, testY   = test_set

    # Get the dataset sizes
    train_size = trainX.get_value(borrow=True).shape[0]
    valid_size = validX.get_value(borrow=True).shape[0]
    test_size  = testX.get_value(borrow=True).shape[0]


    classifier = MLP(
                    rng = np.random.RandomState(SEED),
                    inpt = X,
                    layers = layers,
                    scale = scale
                )
    cost = (
            classifier.negative_log_likelihood(Y) 
            + L2_reg * classifier.L2_sqr # using the L2 regularization
        )

    gparams = [T.grad(cost, param) for param in classifier.params]

    train_model = theano.function(
                 inputs = [index, eta],
                 outputs = cost,
                 updates = [(param, param - eta * gparam) 
                    for param, gparam in zip(classifier.params, gparams)],
                 givens = {
                         X : trainX[index * batch_size : (index + 1) * batch_size],
                         Y : trainY[index * batch_size : (index + 1) * batch_size]
                     }
             )
    
    validate_model = theano.function(
                inputs = [index],
                outputs = classifier.errors(Y),
                givens = {
                         X : validX[index * batch_size : (index + 1) * batch_size],
                         Y : validY[index * batch_size : (index + 1) * batch_size]
                }
            )
    
    test_model = theano.function(
                inputs = [index],
                outputs = classifier.errors(Y),
                givens = {
                         X : testX[index * batch_size : (index + 1) * batch_size],
                         Y : testY[index * batch_size : (index + 1) * batch_size]
                }
            )


    train_error = []
    valid_error = []
    test_error  = []

    # Calculate the number of batches.
    n_train_batches = int(train_size / batch_size)
    n_val_batches = int(valid_size / batch_size)
    n_test_batches = int(test_size / batch_size)

    ANNEAL = 10*train_size # rate at which learning parameter "eta" is reduced as iterations increase ( momentum )
    print("Anneal = {}".format(ANNEAL))
    
    start_time = timeit.default_timer()
    learn_rate = etaVal
    for epoch in xrange(epochs):
        # shuffle data, reset the seed so that trainX and trainY are randomized
        # the same way
        theano_seed = int(np.random.rand()*100)
        theano_rng = RandomStreams(theano_seed)
        trainX = trainX[theano_rng.permutation(n=train_size, size=(1,)),]
        theano_rng = RandomStreams(theano_seed)
        trainY = trainY[theano_rng.permutation(n=train_size, size=(1,)),]
        cost = []
        val_cost = []
        for batch_idx in xrange(n_train_batches):
            cost.append(np.mean(np.asarray([train_model(batch_idx, learn_rate)])))

        # Validation error checked in each epoch
        for val_batch_idx in xrange(n_val_batches): 
            val_cost.append(np.mean(np.asarray([validate_model(val_batch_idx)])))

        train_error.append(np.mean(cost))
        valid_error.append(np.mean(val_cost))

        time_check = timeit.default_timer()
        iteration = (epoch * batch_idx) + batch_idx
        print("epoch={}, mean train cost={}, mean_val_cost = {} time = {} eta={}".format(epoch, train_error[-1], valid_error[-1], (time_check - start_time)/60.0, learn_rate))
        # Search and then converge
        learn_rate = etaVal / ( 1.0 + (iteration*1.0 / ANNEAL))

    return train_error, valid_error

if __name__ == '__main__':
    train_set, valid_set, test_set = DataLoader("../mnist.pkl.gz").get_shared_data()
    eta = 0.01
    costs = []
    epochs = 200
    layers = [784,225,144,10]
    L2_reg = 0.001
    algo = 'mini_batch'
    if algo == 'batch':
        # Batch Gradient Descent
        costs.append(do_gd(eta, epochs, layers, train_set))
    else:
        # Minibatch gradient descent
        train_err, val_err = do_gd(eta, epochs, layers, train_set, 
                valid_set=valid_set, test_set=test_set, batch_size=100, L2_reg=L2_reg)
    print("Final validation error {}".format(val_err[-1]))
    pl.plot(train_err,label="Train Err")
    pl.plot(val_err, label="Val Err")
    pl.legend()
    pl.xlabel("Epochs")
    pl.ylabel("Error")
    pl.title("Train Error vs Validation error")
    pl.savefig("ex4.png")
    pl.show()
