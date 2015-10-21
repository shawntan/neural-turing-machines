import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import controller
import model
import tasks
import random
import math


def make_model(
        input_size=8,
        output_size=8,
        mem_size=128,
        mem_width=20,
        hidden_sizes=[100]):
    P = Parameters()
    ctrl = controller.build(P, input_size, output_size,
                            mem_size, mem_width, hidden_sizes)
    predict = model.build(P, mem_size, mem_width, hidden_sizes[-1], ctrl)
    input_seq = T.matrix('input_sequence')
    [M_curr, weights, output] = predict(input_seq)

    test_fun = theano.function(
        inputs=[input_seq],
        outputs=[weights, output]
    )
    return P, test_fun
