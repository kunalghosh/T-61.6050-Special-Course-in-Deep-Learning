import theano.tensor as T
from logistic_reg import LogisticRegression
from hiddenLayer import HiddenLayer

class MLP(object):
    def __init__(self, rng, inpt, layers, scale=1):
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
                        activation = T.tanh,
                        scale = scale
                    )
            # print (prevLayer, layer)
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
	self.L2_sqr = (
                    sum([(hiddenLayer.W ** 2).sum() for hiddenLayer in self.hidden_layers]) 
                    + (self.logisticRegressionLayer.W ** 2).sum()
                )

        self.negative_log_likelihood = (
                                self.logisticRegressionLayer.negative_log_likelihood
                                )
        self.errors = self.logisticRegressionLayer.get_error
        self.params.extend(self.logisticRegressionLayer.params)
