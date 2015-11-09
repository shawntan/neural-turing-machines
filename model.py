import theano
import theano.tensor as T
import numpy as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import scipy

import head
import feedforward
import ntm
def build(P, input_size, output_size, mem_size, mem_width, controller_size):
    head_count = 1
    P.memory_init = np.random.randn(mem_size,mem_width)  #2 * (np.random.rand(mem_size, mem_width) - 0.5

    weight_init_params = []
    for i in xrange(head_count):
        P['read_weight_init_%d'%i]  = 0.01 * np.random.randn(mem_size)
        P['write_weight_init_%d'%i] = 0.01 * np.random.randn(mem_size)
        weight_init_params.append((P['read_weight_init_%d'%i],P['write_weight_init_%d'%i]))
#        weight_init_params.append((init,init))

    heads_size, head_activations = head.build(
            head_count=head_count,
            mem_width=mem_width,
            shift_width=3
        )
    print "Size of heads:",heads_size

    def controller_activation(X):
        return (head_activations(X[:,:heads_size]),X[:,heads_size:])

    def output_inits(ins,outs):
        init = feedforward.initial_weights(ins,outs)
        init[:,heads_size:] = 0
        return init

    controller = feedforward.build_classifier(
            P, "controller",
            input_sizes=[input_size,mem_width],
            hidden_sizes=[controller_size],
            output_size=heads_size + output_size,
            activation=T.tanh,
            output_activation=controller_activation,
            output_initial_weights=output_inits
        )

    ntm_step = ntm.build(mem_size, mem_width)

    def process(X):
        # input_sequences: batch_size x sequence_length x input_size
        memory_init = P.memory_init / T.sqrt(T.sum(T.sqr(P.memory_init),axis=1,keepdims=True))
        batch_size = X.shape[0]
        batch_size.name = 'batch_size'
        ones = T.ones_like(X[:,0,0])
        batch_memory_init = T.alloc(memory_init,batch_size,mem_size,mem_width)
        batch_memory_init.name = 'batch_memory_init'

        import head
        batch_weight_inits = [
                (
                    head.softmax(r) * ones.dimshuffle(0,'x'),
                    head.softmax(w) * ones.dimshuffle(0,'x')
                ) for r,w in weight_init_params ]

        def step(X,M_prev,*heads):
            X.name = 'x[t]'
            # weights [ batch_size x mem_size ]
            # M_prev  [ batch_size x mem_size x mem_width ]
            weights_prev = zip(heads[0*head_count:1*head_count],
                               heads[1*head_count:2*head_count])
            for r,w in weights_prev:
                r.name = 'read_prev'
                w.name = 'write_prev'


            reads_prev = [ T.sum(r.dimshuffle(0,1,'x') * M_prev,axis=1) 
                                for r,_ in weights_prev ]

            heads, output = controller([X] + reads_prev)
            M_curr, weights_curr = ntm_step(M_prev, heads, weights_prev)

            return [ M_curr ] + \
                   [ r for r,_ in weights_curr ] +\
                   [ w for _,w in weights_curr ] +\
                   [ output ]

        scan_outs, _ = theano.scan(
                step,
                sequences=[X.dimshuffle(1,0,2)],
                outputs_info=[batch_memory_init] +\
                        [ r for r,_ in batch_weight_inits ] +\
                        [ w for _,w in batch_weight_inits ] +\
                        [ None ]
            )
        outputs = scan_outs[-1]
        return outputs.dimshuffle(1,0,2)
    return process

if __name__ == "__main__":
    import feedforward
    import head
    import ntm
    batch_size = 10
    input_size = 10
    output_size = 10
    mem_size = 128
    mem_width = 20
    controller_size = 100
    P = Parameters()

    """
    head_size,head_activations = head.build(1, mem_width, 3)

    controller = feedforward.build_classifier(
            P, "controller",
            input_sizes=[input_size,mem_width],
            hidden_sizes=[20],
            output_size=head_size,
            activation=T.nnet.sigmoid,
            output_activation=head_activations
        )
    ntm_update = ntm.build(mem_size, mem_width)

    weight = theano.shared(np.eye(mem_size)[:batch_size])
    read   = theano.shared(np.random.randn(batch_size,mem_width))
    M      = theano.shared(np.random.randn(batch_size,mem_size,mem_width))

    heads = controller([
        theano.shared(np.random.randn(batch_size,input_size)),
        read
    ])

    M, weight, read = ntm_update(weight,M,heads)
    """

    process = build(P, input_size, output_size, mem_size, mem_width, controller_size)
    output_tape = process(theano.shared(np.random.randn(10,20,10).astype(np.float32))).eval()
    print output_tape.shape
#    print M.eval().shape
