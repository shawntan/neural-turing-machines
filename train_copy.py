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
np.random.seed(1234)
def make_functions(
        input_size, output_size, mem_size, mem_width, hidden_sizes=[100]):

    start_time = time.time()

    input_seqs  = T.btensor3('input_sequences')
    output_seqs = T.btensor3('output_sequences')

    P = Parameters()
    process = model.build(P, 
            input_size, output_size, mem_size, mem_width, hidden_sizes[0])
    outputs = process(T.cast(input_seqs,'float32'))
    outputs.name = 'outputs'
    predicted_seqs = T.nnet.sigmoid(outputs)
    output_length = input_seqs.shape[1] // 2
    
    Y = output_seqs[:,-output_length:,:-2]
    Y_hat = predicted_seqs[:,-output_length:,:-2]

    cross_entropy = T.mean(
            T.nnet.binary_crossentropy((1 - 1e-3) * Y_hat + 1e-3 * 0.5,Y))
    bits_loss = -T.sum(Y * T.log2(Y_hat) + (1 - Y) * T.log2(1 - Y_hat))

    print P.parameter_count()
    params = P.values()
    cost = cross_entropy + 1e-5 * sum(T.sum(T.sqr(w)) for w in params)

    print "Computing gradients",
    grads = T.grad(cost, wrt=params)
    clip_length = 10
    magnitude_sqr = sum(T.sum(T.sqr(g)) for g in grads)
#    grads = [ T.switch(T.gt(magnitude_sqr,T.sqr(clip_length)),
#                        clip_length * g / T.sqrt(magnitude_sqr), g) for g in grads ]
    grads = [ T.clip(g,-10,10) for g in grads ]
    print "Done. (%0.3f s)"%(time.time() - start_time)
    sq_deltas = [ (p.name,d) for p,d in zip(params,grads) ]
    start_time = time.time()
    print "Compiling function",

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                np.isnan(output[0]).any()):
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [(output[0],output[0].shape) for output in fn.outputs]
                #break
                raise Exception('NaN DETECTED.')


    train = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=cross_entropy, #[ d for _,d in sq_deltas ],#
            updates=updates.rmsprop(
                params, grads,
                learning_rate=1e-4
            ),
#            mode=theano.compile.MonitorMode(post_func=detect_nan)
        )

    test = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=bits_loss
        )
    print "Done. (%0.3f s)"%(time.time() - start_time)

    return P, train, test

if __name__ == "__main__":
    from pprint import pprint
    model_out = sys.argv[1]
    width = 8
    max_sequences = 100000
    max_sequence_length = 20
    batch_size = 8

    validation_frequency = 1000

    P, train, test = make_functions(
            input_size=width + 2,
            mem_size=128,
            mem_width=20,
            output_size=width + 2
        )

    test_set = [ tasks.copy(10, l+1, width) for l in xrange(max_sequence_length) ]
    def validation_test():
         return sum(test(i,o) for i,o in test_set) / (10 * len(test_set))
    #P.load('model.pkl')
    best_score = np.inf
    for counter in xrange(max_sequences):
        length = np.random.randint(max_sequence_length) + 1
        i, o = tasks.copy(batch_size, length, width)
        score = train(i, o)
        if np.isnan(score):
            print "NaN"
            exit()
        if (counter + 1) % validation_frequency == 0:
            validation_score = validation_test()
            print validation_score,
            if validation_score < best_score:
                best_score = validation_score
                P.save('model.pkl')
                print "Saving model."
            else:
                print
    print "Best validation score:",best_score
