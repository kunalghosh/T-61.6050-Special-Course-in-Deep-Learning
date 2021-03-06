from __future__ import division
import gzip
import cPickle

import theano
import theano.tensor as T
import numpy as np
import pylab as pl

# ------- DEBUG CODE --------
# def softmax(mat):
#     exp_vals = np.exp(mat)
#     sum_exp = np.sum(exp_vals,axis=1)
#     return np.true_divide(exp_vals,sum_exp)
# 
# def neq(a,b):
#     return np.mean((a == b).astype(int))
# 
# T = np
# T.nnet.softmax = softmax
# np.neq = neq 
# ------- END DEBUG ---------

def get_cost(p_y_given_x,y):
    '''
    Log likelihood cost
    '''
    return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])

def get_p_y_given_x(W,X,b):
    return T.nnet.softmax(T.dot(X,W)+b)

def get_error(y_pred,y):
    return T.mean(T.neq(ypred,y))

def do_simple_gd(train_set,eta,epochs):
    X = T.matrix('X')
    # X.tag.test_value =np.asarray([[1,2,3],[2,3,4],[4,5,6]]) 
    Y = T.ivector('Y')
    # Y.tag.test_value =np.asarray([1,2,1]).astype(np.int32) 
    # Input dimensions
    n_in = 28 * 28
    # Output dimensions
    n_out = 10

    # Get data
    trainX, trainY = train_set

    W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
                ),
            name = 'W',
            borrow = True)
    b = theano.shared(
            value = np.zeros(
                (n_out,),
                dtype = theano.config.floatX
                ),
            name = 'b',
            borrow = True
            )
    cost = get_cost(get_p_y_given_x(W,X,b),Y) 

    gW = T.grad(cost=cost, wrt=W)
    gB = T.grad(cost=cost, wrt=b)

    updates = [
                (W, W - eta * gW),
                (b, b - eta * gB)
            ]

    train_model = theano.function(
                #inputs = [dataX,dataY],
                inputs = [X,Y],
                outputs = cost, 
                updates = updates
                #, 
                #givens = {
                #        X: trainX,
                #        Y: trainY
                #    } 
            )
    cost = np.asarray([train_model(trainX,trainY.astype(np.int32)) for _ in xrange(epochs)])
    print("Eta = {}, Cost Last= {} Mean last 10 Costs = {}".format(eta, cost[-1], np.mean(cost[-10:])))
    return cost[-1]

if __name__ == '__main__':
    # load dataset
    dataset = "mnist.pkl.gz"
    with gzip.open(dataset,"rb") as f: 
        train_set, valid_set, test_set = cPickle.load(f)
    
    # start with eta around 10^-3 * 10
    # iterate over eta * 9, eta * 8,..., eta * 1
    # over iterations Error should reduce
    etas = np.arange(1,21)[::-1]*(np.random.random())
    costs = []
    epochs = 100
    for eta in etas:
        costs.append(do_simple_gd(train_set, eta, epochs=100))

    pl.loglog(etas, costs)
    pl.legend("eta vs costs in log scale")
    pl.xlabel("eta")
    pl.ylabel("cost")
    pl.savefig("02_logistic_gd.png")
    pl.show()
