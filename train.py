import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
import controller
import model
import tasks


def make_accumulate_update(inputs,outputs,parameters,gradients,update_method=updates.adadelta):
	acc = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	count = U.create_shared(np.int32(0))
	acc_update = [ (a,a + g) for a,g in zip(acc,gradients) ] + [ (count,count+1) ]
	acc_gradient = theano.function(
				inputs = inputs,
				outputs = outputs,
				updates = acc_update
			)
	avg_gradient = [ a/count for a in acc ]
	clear_update = [ (a,0.*a) for a,g in zip(acc,parameters) ] + [ (count,0) ]
	train_acc = theano.function(
			inputs=[],
			updates=update_method(parameters,avg_gradient) + clear_update
		)
	return acc_gradient,train_acc


def make_functions(input_size,output_size,mem_size,mem_width,hidden_sizes=[20,20]):
	P = Parameters()
	ctrl = controller.build(P,input_size,output_size,mem_size,mem_width,hidden_sizes)
	predict = model.build(P,mem_size,mem_width,input_size,ctrl)
	
	input_seq = T.matrix('input_sequence')
	output_seq = T.matrix('output_sequence')
	seqs = predict(input_seq)
	output_seq_pred = seqs[-1]
	cost = - T.mean(
			T.sum(
				output_seq * T.log(output_seq_pred) +\
				(1 - output_seq)* T.log(1-output_seq_pred),
				axis=1
			)
		)
	params = P.values()
	grads  = T.grad(cost,wrt=params)

	return make_accumulate_update(
			inputs = [input_seq,output_seq],
			outputs = [cost]+seqs,
			parameters = params,
			gradients  = grads
		)


if __name__ == "__main__":
	input_size = 8
	mem_size   = 128
	mem_width  = 20
	output_size = input_size

	for _ in xrange(30):
		for _ in xrange(5):
			acc_gradient,train_acc = make_functions(input_size,output_size,mem_size,mem_width,hidden_sizes=[100])
			inputs,outputs = tasks.copy(input_size,20)
			[cost,M_curr,read_weight,erase_weight,add_weight,output] = acc_gradient(inputs,outputs)
		train_acc()
