import numpy as np
class FixedMeanStd:
    "fixed mean and standard deviation for normalization for a given environment"
    def __init__(self, env):
        self.envID = env.envID
        if env.envID == 'metapop1':
            self.T = env.T
        self.stored_batch = []
        self.rolloutnum = 0
        self.updateN = 1000 # Number of samples to collect before updating the mean and variance

    def update(self):
        self.stored_batch = []
        self.rolloutnum = 0

    def normalize(self, x):
        # check if there is a variable self.envID
        y = np.asarray(x, dtype=np.float32).copy()
        if 'metapop1' in self.envID: 
            y[-1] =  y[-1] /self.T # only normalize time variable for metapop1
            return y
        else:
            return (y - self.mean) / np.sqrt(self.var + 1e-8)