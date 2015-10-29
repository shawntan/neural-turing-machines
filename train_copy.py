import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import model
import tasks
import sys

np.random.seed(1234)

import time
def make_train(input_size, output_size, mem_size, mem_width, hidden_sizes=[100]):
    start_time = time.time()
    input_seqs  = T.btensor3('input_sequences')
    output_seqs = T.btensor3('output_sequences')

    P = Parameters()
    process = model.build(P, 
            input_size, output_size, mem_size, mem_width, hidden_sizes[0])
    outputs, weight_sums = process(T.cast(input_seqs,'float32'))

    output_length = input_seqs.shape[1] // 2
    predicted_seqs = T.nnet.sigmoid(outputs)
    cross_entropy = T.mean(T.sum(
            T.nnet.binary_crossentropy(
                0.99 * predicted_seqs[:,-output_length:] + 0.01 * 0.5,
                output_seqs[:,-output_length:]
            ),
            axis=(1,2)
        ))
    params = P.values()
    cost = cross_entropy
    print "Computing gradients",
    grads = T.grad(cost, wrt=params)
    print "Done. (%0.3f s)"%(time.time() - start_time)

    sq_deltas = [ (p.name,d) for p,d in zip(params,grads) ]
    start_time = time.time()
    print "Compiling function",
    train = theano.function(
        inputs=[input_seqs, output_seqs],
        outputs=[ d for _,d in sq_deltas ],#cost/output_length, #
        updates=updates.rmsprop(params, grads, learning_rate=1e-4)
    )
    print "Done. (%0.3f s)"%(time.time() - start_time)

    return P, train, params

if __name__ == "__main__":
    from pprint import pprint
    model_out = sys.argv[1]
    width = 7
    P, train, params = make_train(
        input_size=width + 1,
        mem_size=128,
        mem_width=20,
        output_size=width + 1
    )

    max_sequences = 100000
    max_sequence_length = 20
    batch_size = 32
    P.load(model_out)
    for counter in xrange(max_sequences):
        length = np.random.randint(max_sequence_length) + 1
        i, o = tasks.copy(batch_size, length, width)
        score = train(i, o)
        #if np.isnan(score):
        #    print "NaN"
        #    exit()
        #else:
        print score, length
        exit()
#        print { p.name:s for p,s in zip(params,score) }
    P.save(model_out)
