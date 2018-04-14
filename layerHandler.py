import numpy as np

class Layer(object):

    def __init__(self, ):
        pass

    def feedforward(self):
        #compute forward propagation,
        #returns the output
        pass

    def grads_wrt_output(self):
        #compute gradient with respect to back propagation
        pass

    def grads_wrt_weights(self):
        #compute gradient with respect to the weights
        pass

    def update_weights(self):
        #update the weights wrt to the error
        pass

class Activation(Layer):

    def __init__(self):
        pass

class ReluLayer(Activation):
    def __init__(self):



class Affine(Layer):
    def __init__(self,input_dim, output_dim, rng, bias):
        #creates
        self.weights
