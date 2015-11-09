import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import scipy
import head
from theano.printing import Print
from theano_toolkit import ops
def cosine_sim(k, M):
    # k: batch_size x mem_width
    # M: batch_size x mem_size x mem_width
    k = k.dimshuffle(0,'x',1)
    dot_prod = T.sum(k * M,axis=2)
    norm_prod = T.sqrt(T.sum(T.sqr(k),axis=2)) * T.sqrt(T.sum(T.sqr(M),axis=2)) + 1e-5
    batch_sim = dot_prod / norm_prod
    return batch_sim

def build(mem_size, mem_width,
          similarity=cosine_sim,
          shift_width=3):
    shift_conv = scipy.linalg.circulant(np.arange(mem_size)).T[
            np.arange(-(shift_width // 2), (shift_width // 2) + 1)
        ][::-1]

    def log_shift_convolve(log_weight, log_shift):
        # weight: batch_size x mem_size
        # shift:  batch_size x shift_width
        log_shift = log_shift.dimshuffle(0,1,'x')
        log_weight_windows = log_weight[:,shift_conv]
        # batch_size x shift_width x mem_size
        log_shifted_weight = ops.log_sum_exp(log_shift + log_weight_windows,axis=1)
        return log_shifted_weight

    def compute_memory_curr(M_prev, weights, erase_values, add_values):
        # M_prev:     batch_size x mem_size x mem_width
        # weight:     batch_size x mem_size
        # erase_head: batch_size x mem_width
        # add_head:   batch_size x mem_width
        weights    = [ w.dimshuffle(0,1,'x') for w in weights ]

        add   = sum( w * a.dimshuffle(0,'x',1)
                            for w,a in zip(weights,add_values) )
        erase = sum( w * e.dimshuffle(0,'x',1)
                            for w,e in zip(weights,erase_values) )

        M_curr_ = M_prev * (1 - erase)
        M_curr  = M_curr_ + add

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
        score_c = T.addbroadcast(beta,1) * similarity(key, M)
        log_weight_c  = ops.log_softmax(score_c)
        # log_weight_c: batch_size x mem_size


        # 3.3.2 Focusing by Location
        g = (1 - 1e-8) * T.addbroadcast(g,1) + 1e-8 * 0.5
        log_weight_prev = T.log((1 - 1e-8) * weight_prev +\
                1e-8 * (1/T.cast(weight_prev.shape[1],'float32')))
        log_weight_prev.name = 'log_weight_prev'
        log_weight_g = ops.log_add(
                        T.log(g) + log_weight_c,
                        T.log(1 - g) + log_weight_prev
                    )
        log_weight_shifted = log_shift_convolve(log_weight_g, T.log(shift))
        log_weight_shifted.name = 'log_weight_shifted'
        log_weight_sharp = T.addbroadcast(gamma,1) * log_weight_shifted
        weight_curr = ops.softmax(log_weight_sharp)
        return weight_curr

    def ntm_step(M_prev, heads, weights_prev):
        weights_curr = []
        for (read_weight_prev, write_weight_prev), head in zip(weights_prev,heads):
            write_weight_curr = compute_weight_curr(
                            weight_prev=write_weight_prev,
                            M=M_prev,
                            key=head["write_key"],
                            beta=head["write_beta"],
                            g=head["write_g"],
                            shift=head["write_shift"],
                            gamma=head["write_gamma"]
                        )

            read_weight_curr = compute_weight_curr(
                            weight_prev=write_weight_prev,
                            M=M_prev,
                            key=head["read_key"],
                            beta=head["read_beta"],
                            g=head["read_g"],
                            shift=head["read_shift"],
                            gamma=head["read_gamma"]
                        )
            weights_curr.append((read_weight_curr,write_weight_curr))

        M_curr = compute_memory_curr(M_prev,
                weights=[ w for _,w in weights_curr ],
                add_values=[ head["write_add"] for head in heads ],
                erase_values=[ head["write_erase"] for head in heads ]
            )
        return M_curr, weights_curr
    return ntm_step
