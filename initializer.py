import numpy as np

class Initializer(object):

    def __init__(self,rng=np.random.rand(2)):
        self.rng = rng


class GloroInit(Initializer):

    def __init__(self, in, out):
        #ways of computing gloro etc.
        pass
