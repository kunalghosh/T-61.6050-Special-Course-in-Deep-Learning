import gzip
import cPickle as pkl

import theano
import theano.tensor as T
import numpy as np

class DataLoader(object):
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        with gzip.open(self.dataset, 'rb') as f:
            self.train_set, self.valid_set, self.test_set = pkl.load(f)
    
    def get_data(self):
        return (self.train_set,
                self.valid_set,
                self.test_set)

    def __shared_dataset(self, data_xy, Ydtype=None, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        if Ydtype is not None:
            shared_y = T.cast(shared_y, Ydtype)
        return shared_x, shared_y

    def get_shared_data(self):
        Ydtype = 'int32'
        shared = self.__shared_dataset
        return (shared(self.train_set, Ydtype=Ydtype),
                shared(self.valid_set, Ydtype=Ydtype),
                shared(self.test_set,  Ydtype=Ydtype))
