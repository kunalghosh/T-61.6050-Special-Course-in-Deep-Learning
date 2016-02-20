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

class LogisticRegression(object):
    '''
    Refactoring the methods used in ex2 into a class
    '''
    def __init__(self, inpt, n_in, n_out):
        self.inpt = inpt

        # print (n_in, n_out)
        self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    # (n_out, n_in),
                    dtype = theano.config.floatX
                    ),
                name = 'W',
                borrow = True)
        self.b = theano.shared(
                value = np.zeros(
                    (n_out,),
                    # (n_in,),
                    dtype = theano.config.floatX
                    ),
                name = 'b',
                borrow = True
                )
        self.p_y_given_x = self.__get_p_y_given_x()
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        '''
        Log likelihood cost
        '''
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def __get_p_y_given_x(self):
        return T.nnet.softmax(T.dot(self.inpt,self.W)+self.b)
    
    def get_error(self, y):
        return T.mean(T.neq(self.y_pred,y))

def do_gd(train_set,eta,epochs,batch_size=None):
    X = T.matrix('X')
    # X.tag.test_value =np.asarray([[1,2,3],[2,3,4],[4,5,6]]) 
    Y = T.ivector('Y')
    # Y.tag.test_value =np.asarray([1,2,1]).astype(np.int32) 
    # Input dimensions
    n_in = 28 * 28
    # Output dimensions
    n_out = 10
    
    # Index for mini batch
    index = T.lscalar()

    # data set size
    dataset_size = train_set[0].shape[0]
    
    eta = np.asarray(eta, dtype=theano.config.floatX)

    # Get data
    trainX, trainY = train_set
    trainX = theano.shared(np.asarray(trainX, dtype = theano.config.floatX), borrow=True)
    trainY = T.cast(theano.shared(np.asarray(trainY, dtype = theano.config.floatX), borrow = True), 'int32')

    # Constructing the logistic regression classifier
    classifier = LogisticRegression(inpt=X, n_in=n_in, n_out=n_out)

    cost = classifier.negative_log_likelihood(Y) 

    gW = T.grad(cost=cost, wrt=classifier.W)
    gB = T.grad(cost=cost, wrt=classifier.b)

    updates = [
                (classifier.W, classifier.W - eta * gW),
                (classifier.b, classifier.b - eta * gB)
            ]

    # inputs = []
    # givens = {}

    if batch_size == None:
        train_model = theano.function(
                # inputs = [X,Y], 
                inputs = [],
                #batch_size is an argument since it can be used to simulate batch and mini_batch GD
                outputs = cost, 
                updates = updates,
                givens = {
                            X : trainX,
                            Y : trainY
                    }
            )
    else:
        inputs = [index] # index can be atleast 1 and atmost len(train_set)/batch_size
        givens = {
                    X : trainX[index * batch_size : (index + 1) * batch_size],
                    Y : trainY[index * batch_size : (index + 1) * batch_size]
                }
        train_model = theano.function(
                inputs = inputs,
                outputs = cost,
                updates = updates,
                givens = givens
            )


    
    if batch_size == None:
        # cost = np.asarray([train_model(trainX,trainY) for _ in xrange(epochs)])
        cost = np.asarray([train_model() for _ in xrange(epochs)])
    else:
        cost = []
        for _ in xrange(epochs):
            for batch_idx in xrange(int(dataset_size / batch_size)):
                cost.append(np.mean(np.asarray(
                                        [train_model(batch_idx)]
                                        )
                                )
                        )
    # cost = np.asarray([train_model() for _ in xrange(epochs)])
    print("Eta = {}, Cost Last= {} Mean last 10 Costs = {}".format(eta, cost[-1], np.mean(cost[-10:])))
    return cost[-1]

if __name__ == '__main__':
    # load dataset
    dataset = "../mnist.pkl.gz"
    with gzip.open(dataset,"rb") as f: 
        train_set, valid_set, test_set = cPickle.load(f)
    
    # start with eta around 10^-3 * 10
    # iterate over eta * 9, eta * 8,..., eta * 1
    # over iterations Error should reduce
    etas = np.arange(1,21)[::-1]*(np.random.random())
    costs = []
    epochs = 100
    # Batch Gradient Descent
    for eta in etas:
        costs.append(do_gd(train_set, eta, epochs=100))

    # Minibatch gradient descent
    # for eta in etas:
    #     costs.append(do_gd(train_set, eta, epochs=100, batch_size=500))

    pl.loglog(etas, costs)
    pl.legend("eta vs costs in log scale")
    pl.xlabel("eta")
    pl.ylabel("cost")
    pl.savefig("02_logistic_gd.png")
    pl.show()
