import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import controller
import head
import scipy


def cosine_sim(k, M):
    k_unit = k / (T.sqrt(T.sum(k**2)) + 1e-5)
    # T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
    k_unit = k_unit.dimshuffle(('x', 0))
    k_unit.name = "k_unit"
    M_lengths = T.sqrt(T.sum(M**2, axis=1)).dimshuffle((0, 'x'))
    M_unit = M / (M_lengths + 1e-5)
    M_unit.name = "M_unit"
    return T.sum(k_unit * M_unit, axis=1)


def build_step(P, controller,
               controller_size, mem_size, mem_width,
               similarity=cosine_sim,
               shift_width=3,
               no_heads=1):
    shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[
            np.arange(-(shift_width // 2), (shift_width // 2) + 1)
        ][::-1]

    P.memory_init = 2 * (np.random.rand(mem_size, mem_width) - 0.5)
    P.weight_init = np.random.randn(mem_size)

    memory_init = P.memory_init
    weight_init = U.vector_softmax(P.weight_init)

    heads = [head.build(P, h, controller_size, mem_width,
                        mem_size, shift_width) for h in range(no_heads)]

    def build_memory_curr(M_prev, erase_head, add_head, weight):
        weight = weight.dimshuffle((0, 'x'))

        erase_head = erase_head.dimshuffle(('x', 0))
        add_head = add_head.dimshuffle(('x', 0))

        M_erased = M_prev * (1 - (weight * erase_head))
        M_curr = M_erased + (weight * add_head)
        return M_curr

    def build_read(M_curr, weight_curr):
        return T.dot(weight_curr, M_curr)

    def shift_convolve(weight, shift):
        shift = shift.dimshuffle((0, 'x'))
        return T.sum(shift * weight[shift_conv], axis=0)

    def build_head_curr(weight_prev, M_curr, head, input_curr):
        """
        This function is best described by Figure 2 in the paper.
        """
        key, beta, g, shift, gamma, erase, add = head(input_curr)

        # 3.3.1 Focusing b Content
        weight_c = U.vector_softmax(beta * similarity(key, M_curr))
        weight_c.name = "weight_c"

        # 3.3.2 Focusing by Location
        weight_g = g * weight_c + (1 - g) * weight_prev
        weight_g.name = "weight_g"

        weight_shifted = shift_convolve(weight_g, shift)

        weight_sharp = weight_shifted ** gamma
        weight_curr = weight_sharp / T.sum(weight_sharp)

        return weight_curr, erase, add

    def step(input_curr, M_prev, weight_prev):
        read_prev = build_read(M_prev, weight_prev)
        output, controller_hidden = controller(input_curr, read_prev)
        weight_inter, M_inter = weight_prev, M_prev
        for head in heads:
            weight_inter, erase, add = build_head_curr(
                weight_inter, M_inter, head, controller_hidden)
            M_inter = build_memory_curr(M_inter, erase, add, weight_inter)
        weight_curr, M_curr = weight_inter, M_inter
        return M_curr, weight_curr, output
    return step, [memory_init, weight_init, None]


def build(P, mem_size, mem_width, controller_size, ctrl):
    step, outputs_info = build_step(
        P, ctrl, controller_size, mem_size, mem_width)

    def predict(input_sequence):
        outputs, _ = theano.scan(
            step,
            sequences=[input_sequence],
            outputs_info=outputs_info
        )
        return outputs
    return predict
