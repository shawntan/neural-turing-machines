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
    # shift_width x mem_size
    P.memory_init = 2 * (np.random.rand(mem_size, mem_width) - 0.5)
    P.weight_init = np.random.randn(mem_size)

    head_size,head_activations = head.build(
            head_count=1,
            mem_width=mem_width,
            shift_width=3
        )


    def controller_activation(X):
        return (head_activations(X[:,:head_size]),X[:,head_size:])

    controller = feedforward.build_classifier(
            P, "controller",
            input_sizes=[input_size,mem_width],
            hidden_sizes=[controller_size],
            output_size=head_size + output_size,
            activation=T.nnet.sigmoid,
            output_activation=controller_activation
        )

    ntm_update = ntm.build(mem_size, mem_width)

    def process(X):
        # input_sequences: batch_size x sequence_length x input_size
        memory_init = P.memory_init
        weight_init = U.vector_softmax(P.weight_init)
        read_init = T.dot(weight_init,memory_init)

        batch_memory_init = T.alloc(memory_init,X.shape[0],mem_size,mem_width)
        batch_weight_init = T.alloc(weight_init,X.shape[0],mem_size)
        batch_read_init   = T.alloc(read_init,X.shape[0],mem_width)

        def step(X,prev_M,prev_weight,prev_read):
            heads, output = controller([X,prev_read])
            M, weight, read = ntm_update(prev_weight,prev_M,heads)
            return M, weight, read, output
        [batch_M,batch_weight,batch_read,outputs], _ = theano.scan(
                step,
                sequences=[X.dimshuffle(1,0,2)],
                outputs_info=[
                    batch_memory_init,
                    batch_weight_init,
                    batch_read_init,
                    None
                ]
            )
        return outputs.dimshuffle(1,0,2), batch_weight.sum(axis=-1)
    return process

if __name__ == "__main__":
    import feedforward
    import head
    import ntm
    batch_size = 10
    input_size = 10
    output_size = 10
    mem_size = 128
    mem_width = 10
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
