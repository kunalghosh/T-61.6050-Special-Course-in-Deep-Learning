from __future__ import division

import theano
import theano.tensor as T
import numpy as np

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


