import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U


def build(P, id, input_size, mem_width, mem_size, shift_width):

    P["W_%d_key" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_key" % id] = 0. * U.initial_weights(mem_width)
    P["W_%d_shift" % id] = U.initial_weights(input_size, shift_width)
    P["b_%d_shift" % id] = 0. * U.initial_weights(shift_width)

    P["W_%d_beta" % id] = 0. * U.initial_weights(input_size)
    P["b_%d_beta" % id] = 0.
    P["W_%d_gamma" % id] = U.initial_weights(input_size)
    P["b_%d_gamma" % id] = 0.
    P["W_%d_g" % id] = U.initial_weights(input_size)
    P["b_%d_g" % id] = 0.

    P["W_%d_erase" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_erase" % id] = 0. * U.initial_weights(mem_width)
    P["W_%d_add" % id] = U.initial_weights(input_size, mem_width)
    P["b_%d_add" % id] = 0. * U.initial_weights(mem_width)

    def head_params(x):
        # key
        key_t = T.dot(x, P["W_%d_key" % id]) + P["b_%d_key" % id]

        # shift
        shift_t = U.vector_softmax(
            T.dot(x, P["W_%d_shift" % id]) + P["b_%d_shift" % id])
        shift_t.name = "shift_t"

        # scalars
        _beta_t = T.dot(x, P["W_%d_beta" % id]) + P["b_%d_beta" % id]
        _gamma_t = T.dot(x, P["W_%d_gamma" % id]) + P["b_%d_gamma" % id]

        beta_t = T.nnet.softplus(_beta_t)
        gamma_t = T.nnet.softplus(_gamma_t) + 1.
#		beta_t  = (_beta_t  > 0)*_beta_t
#		gamma_t = (_gamma_t > 0)*_gamma_t + 1.
#		beta_t  = T.exp(_beta_t)
#		gamma_t = T.exp(_gamma_t) + 1.

        g_t = T.nnet.sigmoid(T.dot(x, P["W_%d_g" % id]) + P["b_%d_g" % id])

        erase_t = T.nnet.sigmoid(
            T.dot(x, P["W_%d_erase" % id]) + P["b_%d_erase" % id])
        add_t = T.dot(x, P["W_%d_add" % id]) + P["b_%d_add" % id]

        return key_t, beta_t, g_t, shift_t, gamma_t, erase_t, add_t
    return head_params
