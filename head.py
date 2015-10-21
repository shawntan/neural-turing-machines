import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
def softmax(x):
    e_x = T.exp(x - T.max(x,axis=-1,keepdims=True))
    out = e_x / T.sum(e_x,axis=-1,keepdims=True)
    return out

def build(head_count, mem_width, shift_width):
    outputs = [
            ("key",   mem_width,   lambda x: x),
            ("add",   mem_width,   lambda x: x),
            ("erase", mem_width,   T.nnet.sigmoid),
            ("shift", shift_width, softmax),
            ("beta",  1, T.nnet.softplus),
            ("gamma", 1, T.nnet.softplus),
            ("g",     1, T.nnet.sigmoid),
        ]
    head_size = sum(size for _,size,_ in outputs)

    def heads(X):
        # X: batch_size x controller_output_size
        X_grouped = X\
                .reshape((X.shape[0],head_count,head_size))\
                .dimshuffle(1,0,2)

        start_idx = 0
        result = []
        for head_id in xrange(head_count):
            head_params = {}
            for name,size,act in outputs:
                head_params[name] = \
                    act(X_grouped[head_id,:,start_idx:start_idx + size])

                start_idx += size
            result.append(head_params)

        return result
    return head_size,heads

