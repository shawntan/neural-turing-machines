import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import math
import model
import tasks
import sys
import time
from theano.compile.nanguardmode import NanGuardMode
from pprint import pprint
np.random.seed(1234)

width = 8
max_sequences = 100000
max_sequence_length = 20
batch_size = 20
validation_frequency = 100
clip_length = 1


def make_functions(
        input_size, output_size, mem_size, mem_width, hidden_sizes=[100]):

    start_time = time.time()

    input_seqs  = T.btensor3('input_sequences')
    output_seqs = T.btensor3('output_sequences')

    P = Parameters()
    process = model.build(P,
            input_size, output_size, mem_size, mem_width, hidden_sizes[0])
    outputs = process(T.cast(input_seqs,'float32'))
    output_length = (input_seqs.shape[1] - 2) // 2

    Y = output_seqs[:,-output_length:,:-2]
    Y_hat = T.nnet.sigmoid(outputs[:,-output_length:,:-2])

    cross_entropy = T.mean(T.nnet.binary_crossentropy(Y_hat,Y))
    bits_loss = cross_entropy * (Y.shape[1] * Y.shape[2]) / T.log(2)

    params = P.values()

    cost = cross_entropy # + 1e-5 * sum(T.sum(T.sqr(w)) for w in params)

    print "Computing gradients",
    grads = T.grad(cost, wrt=params)
    grads = updates.clip_deltas(grads, np.float32(clip_length))

    print "Done. (%0.3f s)"%(time.time() - start_time)
    start_time = time.time()
    print "Compiling function",
    P_learn = Parameters()

    update_pairs = updates.rmsprop(
                params, grads,
                learning_rate=1e-4,
                P=P_learn
            )

    train = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=cross_entropy,
            updates=update_pairs,
        )

    test = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=bits_loss
        )

    print "Done. (%0.3f s)"%(time.time() - start_time)
    print P.parameter_count()
    return P, P_learn, train, test

if __name__ == "__main__":
    model_out = sys.argv[1]
    P, P_learn, train, test = make_functions(
            input_size=width + 2,
            mem_size=128,
            mem_width=20,
            output_size=width + 2
        )

    test_set = [ tasks.copy(10, l+1, width) for l in xrange(max_sequence_length) ]
    def validation_test():
         return sum(test(i,o) for i,o in test_set) / (10 * len(test_set))

    best_score = validation_test()
    for counter in xrange(max_sequences):
        length = np.random.randint(max_sequence_length) + 1
        i, o = tasks.copy(batch_size, length, width)
        score = train(i, o)

        if (counter + 1) % validation_frequency == 0:
            validation_score = validation_test()
            print validation_score,
            if validation_score < best_score:
                best_score = validation_score
                P.save('model.pkl')
                P_learn.save('learn.pkl')
                print "Saving model."
            else:
                print
    print "Best validation score:",best_score
