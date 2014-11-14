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
import random
import math
def make_accumulate_update(inputs,outputs,parameters,gradients,post_func,update_method=updates.adadelta):
	acc = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	count = U.create_shared(np.int32(0))
	acc_update = [ (a,a + g) for a,g in zip(acc,gradients) ] + [ (count,count+1) ]
	acc_gradient = theano.function( inputs = inputs,
				outputs = outputs,
				updates = acc_update,
				mode=theano.compile.MonitorMode(post_func=post_func).excluding('local_elemwise_fusion', 'inplace')
			)
	avg_gradient = [ a/count for a in acc ]
	clear_update = [ (a,np.zeros(a.get_value().shape,dtype=theano.config.floatX)) for a,g in zip(acc,parameters) ] + [ (count,0) ]
	train_acc = theano.function(
			inputs=[],
			updates=update_method(parameters,avg_gradient) + clear_update
		)
	return acc_gradient,train_acc


def make_functions(input_size,output_size,mem_size,mem_width,hidden_sizes=[20,20]):
	P = Parameters()
	ctrl = controller.build(P,input_size,output_size,mem_size,mem_width,hidden_sizes)
	predict = model.build(P,mem_size,mem_width,hidden_sizes[-1],ctrl)
	
	input_seq = T.matrix('input_sequence')
	output_seq = T.matrix('output_sequence')
	seqs = predict(input_seq)
	output_seq_pred = seqs[-1]
	
	response_length = output_seq.shape[0] / 2
	pred = output_seq_pred[-response_length:]
	# 5e-6 + (1 - 2*5e-6)*
	cross_entropy = T.sum(T.nnet.binary_crossentropy(5e-6 + (1 - 2*5e-6)*output_seq_pred,output_seq),axis=1)
	#sq_error = T.sum((output_seq[-response_length:] - output_seq_pred[-response_length:])**2)
	
	l2 = sum(T.sum(p**2) for p in P.values() if p.name.startswith("W") or p.name.startswith("b"))
	cost = T.sum(cross_entropy) # + 1e-3 * l2
	params = P.values()
	grads  = [ T.clip(g,-100,100) for g in T.grad(cost,wrt=params) ]

	def detect_nan(i, node, fn):
		for output in fn.outputs:
			if np.isnan(output[0]).any() or np.isinf(output[0]).any():
				print '*** NaN detected ***'
				print node
				print 'Inputs : %s' % [input[0] for input in fn.inputs]
				print 'Input shapes : %s' % [input[0].shape for input in fn.inputs]
				print 'Outputs: %s' % [output[0] for output in fn.outputs]
				print 'Output shapes : %s' % [output[0].shape for output in fn.outputs]
				raise Exception('NaN')
				break
	


	return P,make_accumulate_update(
			inputs = [input_seq,output_seq],
			outputs = [T.mean(cross_entropy[-response_length:])]+seqs,
			parameters = params,
			gradients  = grads,
			post_func = detect_nan
		)

	

	
if __name__ == "__main__":
	input_size = 8
	mem_size   = 128
	mem_width  = 20
	output_size = input_size
	counter = 0
	i,o = None,None
	try:
		P,(acc_gradient,train_acc) = make_functions(input_size,output_size,mem_size,mem_width,hidden_sizes=[100])
		[cost,M_curr,weight,output] = 4*[None]

		def run_training(episodes,sequences):
			global i,o
			for eps in xrange(episodes):
				global counter
				length = np.random.randint(int(20*(eps/float(episodes))**1.5)+1) + 1 
				prev_params = { p.name: p.get_value() for p in P.values() }
				costs = []
				for _ in xrange(sequences):
					i,o = tasks.copy(input_size,length)
					cost,_,_,outputs = acc_gradient(i,o)
					costs.append(cost)
					counter += 1
				avg_cost = np.mean(costs)
				print length,avg_cost
				
				train_acc()
				for k,v in prev_params.items():
					if np.sum(P[k].get_value() - v) == 0:
						print k

		#P.load('rmsprop.mdl')
		run_training(75000,1)
		P.save('increasing.mdl')
	except Exception as exp:
		print exp
		print i
		print o
