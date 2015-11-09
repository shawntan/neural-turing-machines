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

import ipdb
def safe_softplus(x):
    return T.switch(x > 10,x,T.nnet.softplus(x))
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
    output_length = (input_seqs.shape[1] - 2) // 2
    Y = output_seqs[:,-output_length:,:-2]
    Y_hat = T.nnet.sigmoid(outputs[:,-output_length:,:-2])
    log_predicted_seqs = -safe_softplus(-outputs[:,-output_length:,:-2])
    log_neg_predicted_seqs = -safe_softplus(outputs[:,-output_length:,:-2])


    cross_entropy = -T.mean(Y * log_predicted_seqs + (1 - Y) * log_neg_predicted_seqs)
    bits_loss = -T.sum(Y * log_predicted_seqs + (1 - Y) * log_neg_predicted_seqs) # -T.sum(Y * T.log2(Y_hat + 1e-8) + (1 - Y) * T.log2(1 - Y_hat + 1e-8))

    print P.parameter_count()
    params = P.values()
    cost = cross_entropy + 1e-5 * sum(T.sum(T.sqr(w)) for w in params)

    print "Computing gradients",
    grads = T.grad(cost, wrt=params)
    grads_ = grads
    names_ = [p.name for p in params]
    clip_length = 10
#    magnitude_sqr = sum(T.sum(T.sqr(g)) for g in grads)
#    grads = [ T.switch(T.gt(magnitude_sqr,T.sqr(clip_length)),
#                        clip_length * g / T.sqrt(magnitude_sqr), g) for g in grads ]
    grads = [ T.clip(g,-clip_length,clip_length) for g in grads ]
    print "Done. (%0.3f s)"%(time.time() - start_time)
    sq_deltas = [ (p.name,d) for p,d in zip(params,grads) ]
    start_time = time.time()
    print "Compiling function",

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if not isinstance(output[0], np.random.RandomState):
                if np.isnan(output[0]).any() or\
                     np.isinf(output[0]).any():
                    if isinstance(node.op,theano.sandbox.cuda.basic_ops.GpuAllocEmpty) or \
                            isinstance(node.op,theano.tensor.basic.AllocEmpty) or \
                        isinstance(node.op,theano.sandbox.gpuarray.subtensor.GpuIncSubtensor) or \
                        isinstance(node.op,theano.tensor.IncSubtensor):
                        print "AllocEmpty complain."
                        pass
                    else:
                        if np.isnan(output[0]).any(): print '*** NaN detected ***'
                        if np.isinf(output[0]).any(): print '*** inf detected ***'
                        theano.printing.debugprint(node)
                        print 'Inputs : %s' % [input[0] for input in fn.inputs]
                        print 'Outputs: %s' % [output[0] for output in fn.outputs]
                        #break
                        raise Exception('NaN DETECTED.')

    P_learn = Parameters()
    train = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=[cross_entropy] + [ T.sqrt(T.sum(T.sqr(g))) for g in grads_ ],
            updates=updates.rmsprop(
                params, grads,
                learning_rate=1e-4,
                P=P_learn
            ),
#            mode=theano.compile.MonitorMode(post_func=detect_nan)
        )

    test = theano.function(
            inputs=[input_seqs, output_seqs],
            outputs=bits_loss
        )
    run = theano.function(
            inputs=[input_seqs,output_seqs], outputs=\
                [Y,Y * log_predicted_seqs + (1 - Y) * log_neg_predicted_seqs],
            on_unused_input='warn'
            )
    print "Done. (%0.3f s)"%(time.time() - start_time)

    return P, P_learn, train, test, run, names_

if __name__ == "__main__":
    from pprint import pprint
    model_out = sys.argv[1]
    width = 8
    max_sequences = 100000
    max_sequence_length = 20
    batch_size = 16

    validation_frequency = 1000

    P, P_learn, train, test, run, names_ = make_functions(
            input_size=width + 2,
            mem_size=128,
            mem_width=20,
            output_size=width + 2
        )

    test_set = [ tasks.copy(10, l+1, width) for l in xrange(max_sequence_length) ]
    def validation_test():
         return sum(test(i,o) for i,o in test_set) / (10 * len(test_set))
    best_score = validation_test()
    i =  np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.int8)
    o =  np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]]], dtype=np.int8)
#    P.load('model.pkl')
#    P_learn.load('learn.pkl')
    print run(i,o)

    nan_counter = 0
    for counter in xrange(max_sequences):
        length = np.random.randint(max_sequence_length) + 1
        i, o = tasks.copy(batch_size, length, width)
        score = train(i, o)
#        pprint(zip(names_,score))
        if np.isnan(score[0]):
            print "NaN"
            nan_counter += 1
            if nan_counter < 20:
                P.load('model.pkl')
                P_learn.load('learn.pkl')
                pprint(zip(names_,score[1:]))
            else:
                print "Too many nans."
                exit()
#            ipdb.set_trace()

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
