import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
def softmax(x):
    e_x = T.exp(x - T.max(x,axis=-1,keepdims=True))
    out = e_x / T.sum(e_x,axis=-1,keepdims=True)
    return out

def build(head_count, mem_width, shift_width):
    outputs = [ ("write_add",   mem_width, T.tanh),
                ("write_erase", mem_width, T.nnet.sigmoid),
                ("write_key",   mem_width, T.tanh), # change to T.tanh?
                ("read_key",    mem_width, T.tanh),
                ("write_shift", shift_width, softmax),
                ("write_beta",  1, T.nnet.softplus),
                ("write_gamma", 1, lambda x:T.nnet.softplus(x) + 1),
                ("write_g",     1, T.nnet.sigmoid),
                ("read_shift",  shift_width, softmax),
                ("read_beta",   1, T.nnet.softplus),
                ("read_gamma",  1, lambda x:T.nnet.softplus(x) + 1),
                ("read_g",      1, T.nnet.sigmoid), ]
    head_size = sum(w for _,w,_ in outputs)
    total_size = head_size * head_count

    def heads(X):
        # X: batch_size x controller_output_size
        X_grouped = X\
                .reshape((X.shape[0],head_count,head_size))\
                .dimshuffle(1,0,2)

        result = []
        for head_id in xrange(head_count):
            head_params = {}
            start_idx = 0
            for name,size,act in outputs:
                head_params[name] = \
                    act(X_grouped[head_id,:,start_idx:start_idx + size])
                start_idx += size
            result.append(head_params)
        return result
    return total_size,heads
