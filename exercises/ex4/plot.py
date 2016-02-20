import pylab as pl
import numpy as np


scales = ["l2_0","l2_0.01","l2_0.001","gauss_100"]

for scale in scales:
    error, time = zip(*np.genfromtxt("train_val_{}.dat".format(scale)))
    pl.plot(error, label="{}".format(scale))
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("Validation Error")
pl.title("Validation Error with different regularizations.")
pl.savefig("validationError_regularization.png")
# pl.title("Scale = {}".format(scale))
# pl.savefig("trainError_time_{}.png".format(scale))
pl.show()

