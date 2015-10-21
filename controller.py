import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import controller
import model
import head
from collections import namedtuple
#from theano_toolkit.parameters import Parameters


def build(P, input_size, output_size, mem_size, mem_width, layer_sizes):
    """
    Create controller function for use during scan op
    """

    P.W_input_hidden = U.initial_weights(input_size, layer_sizes[0])
    P.W_read_hidden = U.initial_weights(mem_width, layer_sizes[0])
    P.b_hidden_0 = 0. * U.initial_weights(layer_sizes[0])

    hidden_weights = []
    for i in xrange(len(layer_sizes) - 1):
        P["W_hidden_%d" %
            (i + 1)] = U.initial_weights(layer_sizes[i], layer_sizes[i + 1])
        P["b_hidden_%d" % (i + 1)] = 0. * U.initial_weights(layer_sizes[i + 1])
        hidden_weights.append(
            (P["W_hidden_%d" % (i + 1)], P["b_hidden_%d" % (i + 1)]))

    P.W_hidden_output = 0. * U.initial_weights(layer_sizes[-1], output_size)
    P.b_output = 0. * U.initial_weights(output_size)

    def controller(input_t, read_t):
        #		print "input_t",input_t.type
        prev_layer = hidden_0 = T.tanh(
            T.dot(input_t, P.W_input_hidden) +
            T.dot(read_t, P.W_read_hidden) +
            P.b_hidden_0
        )

#		print "input",read_t.type,input_t.type
#		print "weights",P.W_input_hidden.type,P.W_read_hidden.type,P.b_hidden_0.type
#		print "layer", hidden_0.type
        for W, b in hidden_weights:
            prev_layer = T.tanh(T.dot(prev_layer, W) + b)

        fin_hidden = prev_layer
        output_t = T.nnet.sigmoid(
            T.dot(fin_hidden, P.W_hidden_output) + P.b_output)

        return output_t, fin_hidden
    return controller
