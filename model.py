import theano
import theano.tensor as T
import numpy as np

import head
import feedforward
import ntm


def build(P, input_size, output_size, mem_size, mem_width, controller_size):
    head_count = 1
    P.memory_init = np.random.randn(mem_size, mem_width)

    weight_init_params = []
    for i in xrange(head_count):
        P['read_weight_init_%d' % i] = 0.01 * np.random.randn(mem_size)
        P['write_weight_init_%d' % i] = 0.01 * np.random.randn(mem_size)
        weight_init_params.append((P['read_weight_init_%d' % i],
                                   P['write_weight_init_%d' % i]))
#        weight_init_params.append((init,init))

    heads_size, head_activations = head.build(
            head_count=head_count,
            mem_width=mem_width,
            shift_width=3
        )
    print "Size of heads:", heads_size

    def controller_activation(X):
        return (head_activations(X[:, :heads_size]), X[:, heads_size:])

    def output_inits(ins, outs):
        init = feedforward.initial_weights(ins, outs)
        init[:, heads_size:] = 0
        return init

    controller = feedforward.build_classifier(
            P, "controller",
            input_sizes=[input_size, mem_width],
            hidden_sizes=[controller_size],
            output_size=heads_size + output_size,
            activation=T.tanh,
            output_activation=controller_activation,
            output_initial_weights=output_inits
        )

    ntm_step = ntm.build(mem_size, mem_width)

    def process(X):
        # input_sequences: batch_size x sequence_length x input_size
        memory_init = P.memory_init / T.sqrt(T.sum(T.sqr(P.memory_init),
                                                   axis=1, keepdims=True))
        batch_size = X.shape[0]
        batch_size.name = 'batch_size'
        ones = T.ones_like(X[:, 0, 0])
        batch_memory_init = T.alloc(memory_init, batch_size, mem_size, mem_width)
        batch_memory_init.name = 'batch_memory_init'

        import head
        batch_weight_inits = [
                (
                    head.softmax(r) * ones.dimshuffle(0, 'x'),
                    head.softmax(w) * ones.dimshuffle(0, 'x')
                ) for r, w in weight_init_params]

        def step(X, M_prev, *heads):
            X.name = 'x[t]'
            # weights [ batch_size x mem_size ]
            # M_prev  [ batch_size x mem_size x mem_width ]
            weights_prev = zip(heads[0*head_count:1*head_count],
                               heads[1*head_count:2*head_count])
            for r, w in weights_prev:
                r.name = 'read_prev'
                w.name = 'write_prev'

            reads_prev = [T.sum(r.dimshuffle(0, 1, 'x') * M_prev, axis=1)
                          for r, _ in weights_prev]

            heads, output = controller([X] + reads_prev)
            M_curr, weights_curr = ntm_step(M_prev, heads, weights_prev)

            return [M_curr] + \
                   [r for r, _ in weights_curr] +\
                   [w for _, w in weights_curr] +\
                   [output]

        scan_outs, _ = theano.scan(
                step,
                sequences=[X.dimshuffle(1, 0, 2)],
                outputs_info=[batch_memory_init] +
                             [r for r, _ in batch_weight_inits] +
                             [w for _, w in batch_weight_inits] +
                             [None]
            )
        outputs = scan_outs[-1]
        return outputs.dimshuffle(1, 0, 2)
    return process
