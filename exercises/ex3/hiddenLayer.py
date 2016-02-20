import numpy as np

import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, inpt, n_in, n_out, W=None, b=None, activation=T.tanh, scale=1):
        '''
        activation : activation used in the hidden Layer
                     Linear unless specified, expects a function.
        weight     : For all functions, tanh weight initialization for sigmoid
                     4 * the weight.
        bias       : Initialized to all zeros
        n_in       : Number of nodes in the previous layer.
        n_out      : Number of nodes in the next layer
        rng        : Random number generator
        '''
        self.inpt = inpt
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        # initialization for tanh as described by [Xavier 10]
                        low  = -np.sqrt(6.0/(n_in + n_out)),
                        high = np.sqrt(6.0/(n_in + n_out)),
                        size = (n_in, n_out)
                        )
                    , dtype = theano.config.floatX
                    )
            if activation == theano.tensor.nnet.sigmoid:
                # Initialization of weghts for sigmoid as described by [Xavier 10]
                # 4 * weights for tanh
                W_values = W_values * 4
            W = theano.shared(value=W_values*scale, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,) ,dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow=True)
        self.W = W
        self.b = b
        
        linear_output = T.dot(self.inpt,self.W) + self.b
        # return linear output unless activatio is specified.
        self.output = (
                    linear_output if activation is None
                    else activation(linear_output)
                )
        # parameters of the model
        self.params = [self.W, self.b]

