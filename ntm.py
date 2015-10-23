import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import scipy
import head

def cosine_sim(k, M):
    # k: batch_size x mem_width
    # M: batch_size x mem_size x mem_width
    k_unit = k / (T.sqrt(T.sum(T.sqr(k),axis=1,keepdims=True)) + 1e-5)
    M_unit = M / (T.sqrt(T.sum(T.sqr(M),axis=2,keepdims=True)) + 1e-5)
    batch_sim = T.sum(k_unit.dimshuffle(0,'x',1) * M_unit, axis=2)
    return batch_sim

def build(mem_size, mem_width,
          similarity=cosine_sim,
          shift_width=3):
    shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[
            np.arange(-(shift_width // 2), (shift_width // 2) + 1)
        ][::-1]

    def shift_convolve(weight, shift):
        # weight: batch_size x mem_size
        # shift:  batch_size x shift_width
        log_shift = T.log(shift).dimshuffle(0,1,'x')
        log_weight_windows = weight[:,shift_conv]
        # batch_size x shift_width x mem_size
        shifted_weight = T.sum(
                T.exp(log_shift + log_weight_windows),
                axis=1
            )
        return shifted_weight

    def compute_memory_curr(M_prev, erase_head, add_head, weight):
        # M_prev:     batch_size x mem_size x mem_width
        # weight:     batch_size x mem_size
        # erase_head: batch_size x mem_width
        # add_head:   batch_size x mem_width
        weight     = weight.dimshuffle(0,1,'x')
        erase_head = erase_head.dimshuffle(0,'x',1)
        add_head   = add_head.dimshuffle(0,'x',1)

        M_erased = M_prev * (1 - (weight * erase_head))
        M_curr   = M_erased + (weight * add_head)

        # output: batch_size x mem_size x mem_width
        return M_curr

    def compute_read(M_curr, weight):
        # M_curr:   batch_size x mem_size x mem_width
        # weight:   batch_size x mem_size

        # output: batch_size x mem_width
        return T.sum(
                weight.dimshuffle(0,1,'x') * M_curr,
                axis=1
            )

    def compute_weight_curr(weight_prev, M, key, beta, g, shift, gamma):
        """
        This function is best described by Figure 2 in the paper.
        """
        # 3.3.1 Focusing b Content
        weight_c_ = T.addbroadcast(beta,1) * similarity(key, M)
        weight_c_ = T.exp(weight_c_)
        weight_c = weight_c_ / T.sum(weight_c_,axis=1,keepdims=True)

        # 3.3.2 Focusing by Location
        g = T.addbroadcast(g,1)
        weight_g = g * weight_c + (1 - g) * weight_prev

        weight_shifted = shift_convolve(weight_g, shift)
        log_weight_sharp = T.addbroadcast(gamma,1) * T.log(weight_shifted)
        weight_curr = head.softmax(log_weight_sharp) 

        return weight_curr

    def ntm_step(weight_prev,M_prev,heads):
        read_prev = compute_read(M_prev, weight_prev)
        weight_inter, M_inter = weight_prev, M_prev
        for head in heads:
            erase = head["erase"]
            add   = head["add"]
            weight_inter = compute_weight_curr(
                    weight_prev=weight_inter,
                    M=M_inter,
                    key=head["key"],
                    beta=head["beta"],
                    g=head["g"],
                    shift=head["shift"],
                    gamma=head["gamma"]
                )
            M_inter = compute_memory_curr(M_inter, erase, add, weight_inter)
        weight_curr, M_curr = weight_inter, M_inter
        return M_curr, weight_curr, read_prev
    return ntm_step
