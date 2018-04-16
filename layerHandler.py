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
        pass

class SigmoidLayer(Activation):
    def __init__(self):
        pass

    def feedforward(self, inputs):
        return 1.0 / (1.0 + np.exp(-inputs))

    def backpropagate(self, inputs, outputs, grads_wrt_outputs):
        return outputs * (1.0 - outputs)

class TanhLayer(Layer):
    """Layer implementing an element-wise hyperbolic tangent transformation."""

    def feedforward(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = tanh(x)`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.tanh(inputs)

    def backpropagate(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (1. - outputs**2) * grads_wrt_outputs

    def __repr__(self):
        return 'TanhLayer'


class Affine(Layer):
    def __init__(self,input_dim, output_dim, Initializer_obj, bias_obj):
        #creates
        self.weights = Initializer_obj(input_dim, output_dim)# returns output_dim x input_dim matrix
        self.bias = bias_obj(output_dim)

    def feedforward(self, inputs):
        #propagate the output forward
        return self.weights.dot(inputs.T).T + self.bias

    def backpropagate(self, inputs,output,grads_wrt_outputs):
        #find the gradient with respect to the input
        return grads_wrt_outputs.dot(self.weights)


    def grads_wrt_params(self,inputs, grads_wrt_outputs):
        grad_wrt_weights = np.dot(grads_wrt_outputs.T, inputs)
        grad_wrt_biases = np.sum(grads_wrt_outputs,axis=0)
        return (grad_wrt_weights, grad_wrt_biases)
