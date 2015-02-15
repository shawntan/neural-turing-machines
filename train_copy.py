import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import controller
import model
import tasks
import sys
np.random.seed(1234)

def make_train(input_size,output_size,mem_size,mem_width,hidden_sizes=[100]):
	P = Parameters()
	ctrl = controller.build(P,input_size,output_size,mem_size,mem_width,hidden_sizes)
	predict = model.build(P,mem_size,mem_width,hidden_sizes[-1],ctrl)
	
	input_seq = T.matrix('input_sequence')
	output_seq = T.matrix('output_sequence')
	seqs = predict(input_seq)
	output_seq_pred = seqs[-1]
	cross_entropy = T.sum(T.nnet.binary_crossentropy(5e-6 + (1 - 2*5e-6)*output_seq_pred,output_seq),axis=1)
	params = P.values()
	l2 = T.sum(0)
	for p in params:
		l2 = l2 + (p ** 2).sum()
	cost = T.sum(cross_entropy) + 1e-3*l2
	grads  = [ T.clip(g,-100,100) for g in T.grad(cost,wrt=params) ]
	
	train = theano.function(
			inputs=[input_seq,output_seq],
			outputs=cost,
			updates=updates.adadelta(params,grads)
		)

	return P,train

if __name__ == "__main__":
	model_out = sys.argv[1]

	P,train = make_train(
		input_size = 8,
		mem_size   = 128,
		mem_width  = 20,
		output_size = 8
	)

	max_sequences = 100000
	patience = 20000
	patience_increase = 3
	improvement_threshold = 0.995
	best_score = np.inf
	test_score = 0.
	score = None
	alpha = 0.95
	for counter in xrange(max_sequences):
		length = np.random.randint(int(20 * (min(counter,50000)/float(50000))**2) +1) + 1
		i,o = tasks.copy(8,length)
		if score == None: score = train(i,o)
		else: score = alpha * score + (1 - alpha) * train(i,o)
		print "round:", counter, "score:", score
		if score < best_score:
			# improve patience if loss improvement is good enough
			if score < best_score * improvement_threshold:
				patience = max(patience, counter * patience_increase)
			P.save(model_out)
			best_score = score
		
		if patience <= counter: break


