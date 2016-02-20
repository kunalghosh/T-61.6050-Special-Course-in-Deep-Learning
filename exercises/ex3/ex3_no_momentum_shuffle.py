import theano
import theano.tensor as T
from theano.printing import pydotprint

import numpy as np
from logistic_reg import LogisticRegression
from hiddenLayer import HiddenLayer
from data_loader import DataLoader
import pylab as pl

class MLP(object):
    def __init__(self, rng, inpt, layers):
        '''
        len(layers) must be atleast 3 
        input -> hidden -> output 
        but it can be more as well.
        '''
        self.hidden_layers = []
        old_inpt = inpt
        self.params = []
        for prevLayer, layer in zip(layers[:-1],layers[1:-1]):
            # if layers = [784,225,144,10]
            # then we want [(784,225) (225,144)]
            hiddenLayer = HiddenLayer(
                        rng = rng,
                        inpt = old_inpt,
                        n_in = prevLayer,
                        n_out = layer,
                        activation = T.tanh
                    )
            print (prevLayer, layer)
            old_inpt = hiddenLayer.output
            self.hidden_layers.append(hiddenLayer)
            self.params.extend(hiddenLayer.params)
        
        self.logisticRegressionLayer = LogisticRegression(
                                        inpt = self.hidden_layers[-1].output,
                                        # n_in = layers[-1],
                                        n_in = layers[-2],
                                        n_out = layers[-1] 
                                        # n_out = layers[-2] 
                                    )
        self.negative_log_likelihood = (
                                self.logisticRegressionLayer.negative_log_likelihood
                                )
        self.errors = self.logisticRegressionLayer.get_error
        self.params.extend(self.logisticRegressionLayer.params)

def do_gd(train_set, eta, epochs, layers, batch_size=None):
    X = T.matrix('X')
    Y = T.ivector('Y')
    index = T.lscalar()

    n_in = layers[0]
    n_out = layers[-1]

    trainX, trainY = train_set
    if batch_size is not None:
        dataset_size = trainX.get_value(borrow=True).shape[0]

    eta = np.asarray(eta, dtype=theano.config.floatX)

    classifier = MLP(
                    rng = np.random.RandomState(5381),
                    inpt = X,
                    layers = layers
                )
    cost = classifier.negative_log_likelihood(Y)

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - eta * gparam) 
                for param, gparam in zip(classifier.params, gparams)
            ]
    if batch_size == None:
        train_model = theano.function(
                    inputs = [],
                    outputs = cost,
                    updates = updates,
                    givens = {
                            X : trainX,
                            Y : trainY
                        }
                )
    else:
        train_model = theano.function(
                    inputs = [index],
                    outputs = cost,
                    updates = updates,
                    givens = {
                            X : trainX[index * batch_size : (index + 1) * batch_size],
                            Y : trainY[index * batch_size : (index + 1) * batch_size]
                        }
                )

    # pydotprint(train_model,'./test.png')
    # d3v.d3viz(train_model,'./test.html')
    if batch_size == None:
        cost = np.asarray([train_model() for _ in xrange(epochs)])
    else:
        cost = []
        n_batches = int(dataset_size / batch_size)
        for epoch in xrange(epochs):
            for batch_idx in xrange(n_batches):
                cost.append(np.mean(np.asarray([train_model(batch_idx)])))
            print("epoch = {} mean cost = {} last cost = {}".format(epoch, np.mean(cost[-n_batches:]), cost[-1]))

    print("Eta = {}, Cost Last= {} Mean last 10 Costs = {}".format(
            eta, cost[-1], np.mean(cost[-10:])) 
         )
    return np.mean(cost[-10:])

if __name__ == '__main__':
    train_set, valid_set, test_set = DataLoader("../mnist.pkl.gz").get_shared_data()
    eta = 0.01
    costs = []
    epochs = 1000
    layers = [784,225,144,10]
    algo = 'mini_batch'
    if algo == 'batch':
        # Batch Gradient Descent
        costs.append(do_gd(train_set, eta, epochs=epochs, layers=layers))
    else:
        # Minibatch gradient descent
        costs = do_gd(train_set, eta, epochs=epochs, layers=layers, batch_size=500)

    pl.loglog(etas, costs)
    pl.legend("eta vs costs in log scale")
    pl.xlabel("eta")
    pl.ylabel("cost")
    pl.savefig("ex3-{}.png".format(algo))
    pl.show()
