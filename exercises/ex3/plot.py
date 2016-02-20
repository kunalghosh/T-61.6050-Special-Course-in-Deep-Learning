import pylab as pl
import numpy as np


scales = [0.01, 0.1, 1, 10, 100]

for scale in scales:
    error, time = zip(*np.genfromtxt("trainError_time_{}.csv".format(scale)))
    pl.plot(time, error, label="{}".format(scale))
pl.legend()
pl.xlabel("Time in minutes")
pl.ylabel("Training Error")
pl.title("Training Error with different scales of initialization of Weight vector.")
pl.savefig("trainingError_time.png")
# pl.title("Scale = {}".format(scale))
# pl.savefig("trainError_time_{}.png".format(scale))
pl.show()

