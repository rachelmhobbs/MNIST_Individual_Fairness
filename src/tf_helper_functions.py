import tensorflow as tf
import numpy as np

def weights(shape, name="weights"):
        '''
        Creates the filters and bias' for the convolutions/pooling layers of the inception_module

        Parameters:
            shape: The shape of the weight tensor.
                    For dense_layers, this should be [num_inputs_to_layer, num_outputs_of_layer].
                    Note: Number of outputs for a layer is equivalent to the number of neurons in that layer.
        Returns:
            The tensor weight variable
        '''
        return(tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name))

def bias(shape, initial_value=0.1, name="bias"):
    '''
    Create a bias variable. Note it is initalized by default to 0.1

    Parameters:
        shape: The shape of the bias tensor. This should be a 1D shape equal to the number of outputs of
                a layer.
        initial_value: The initial value of all the bias's

    Returns:
           The tensor bias variable
    '''
    return(tf.Variable(tf.constant(initial_value, shape=shape, ), name=name))

def dense_layer(inputs, weights, bias, activation=None, name="dense_layer"):
    '''
    Build a fully connected dense neuron layer.

    Parameters:
        inputs: An input tensor with shape [batch_size, num_inputs]
        weights: A tensor with shape [num_inputs, num_outputs]
        bias: A tensor with shape [num_outputs]
        activation: A valid tensorflow activation function.(EX. tf.nn.relu())
    Returns:
        If activation is none, return the output of Wx+b. Else if activation is set, return
        the activation function with input Wx+b.
    '''
    output = tf.matmul(inputs, weights) + bias
    if activation is None:
        return(output)
    return(activation(output, name=name))

def lipschitz_projection(weights, p_norm, lamda):
    '''
    Projection function on a weights matrix to constrain the lipschitz constant.

    Parameters:
        weights: The weights tenosr of a neuron layer
        p_norm: The The distance norm to be computed for the weights
        lamda: Hyperparameter to control the upper bound of the Lipschitz constant

    Returns:
        Assignment operation to update the weights tensor.
        weights_norm: The p_norm of the weights tensro
    '''
    one_scalar = tf.constant(1, dtype=tf.float32)
    lamda = tf.constant(lamda, dtype=tf.float32)

    #compute the p_norm of the weight matrix
    weights_norm = tf.norm(weights, ord=p_norm)
    weight_bound = tf.divide(weights_norm, lamda)
    projection_parameter = tf.divide(one_scalar, tf.maximum(one_scalar, weight_bound))
    weights_projection = tf.multiply(projection_parameter, weights)

    return(tf.assign(weights, weights_projection), weights_norm)
