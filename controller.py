import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def make_controller(P,input_size,mem_size,mem_width,layer_sizes):
	"""
	Create controller function for use during scan op
	"""

	P.W_input_hidden     = U.initial_weights(input_size,layer_sizes[0])
	P.W_mem_width_hidden = U.initial_weights(mem_width, layer_sizes[0])
	P.b_hidden_0 = U.initial_weights(layer_sizes[0])
	hidden_weights = []
	for i in xrange(len(layer_sizes)-1):
		P["W_hidden_%d"%(i+1)] = U.initial_weights(layer_sizes[i],layer_sizes[i+1])
		P["b_hidden_%d"%(i+1)] = U.initial_weights(layer_sizes[i+1])
		hidden_weights.append((P["W_hidden_%d"%(i+1)],P["b_hidden_%d"%(i+1)]))
	P.W_hidden_erase = U.initial_weights(layer_sizes[-1],mem_width)
	P.W_hidden_add   = U.initial_weights(layer_sizes[-1],mem_width)
	P.W_hidden_key   = U.initial_weights(layer_sizes[-1],mem_width)
	P.W_hidden_shift = U.initial_weights(layer_sizes[-1],mem_size)
	P.b_erase = U.initial_weights(mem_width)
	P.b_add   = U.initial_weights(mem_width)
	P.b_key   = U.initial_weights(mem_width)
	P.b_shift = U.initial_weights(mem_size)

	P.W_hidden_beta  = U.initial_weights(layer_sizes[-1])
	P.W_hidden_gamma = U.initial_weights(layer_sizes[-1])
	P.W_hidden_g     = U.initial_weights(layer_sizes[-1])
	P.b_beta  = 0.0
	P.b_gamma = 0.0
	P.b_g     = 0.0

	def controller(input_t,read_t):
		prev_layer = hidden_0 = T.nnet.sigmoid(
				T.dot(input_t,P.W_input_hidden) +\
				T.dot(read_t,P.W_mem_width_hidden) +\
				P.b_hidden_0
			)
		for W,b in hidden_weights:
			prev_layer = T.nnet.sigmoid(T.dot(prev_layer,W) + b)
		fin_hidden = prev_layer
		
		erase_t = T.nnet.sigmoid(T.dot(fin_hidden,P.W_hidden_erase) + P.b_erase)
		add_t   = T.nnet.sigmoid(T.dot(fin_hidden,P.W_hidden_add)   + P.b_add)
		key_t   = T.nnet.sigmoid(T.dot(fin_hidden,P.W_hidden_key)   + P.b_key)

		# shift
		shift_t = T.nnet.softmax(
			T.dot(
				fin_hidden.reshape((1,fin_hidden.shape[0])),
				P.W_hidden_shift
			) + P.b_shift
		)[0]
		shift_t.name = "shift_t"

		# scalars
		_beta_t  = T.dot(fin_hidden,P.W_hidden_beta) + P.b_beta
		_gamma_t = T.dot(fin_hidden,P.W_hidden_gamma) + P.b_gamma
		beta_t  = (_beta_t > 0) * _beta_t
		gamma_t = (_gamma_t >= 1) * _gamma_t
		g_t = T.dot(fin_hidden,P.W_hidden_g) + P.b_g

		return erase_t,add_t,key_t,shift_t,beta_t,gamma_t,g_t
	return controller

if __name__ == "__main__":
	P = Parameters()
	controller = make_controller(P,6,100,6,[20,20])
	i_t = T.vector('i_t')
	r_t = T.vector('r_t')
	f = theano.function(
			inputs = [i_t,r_t],
			outputs = controller(i_t,r_t)
		)
	output = f(np.zeros((6,),dtype=np.float32),np.zeros((6,),dtype=np.float32))
	keys   = [ "erase_t","add_t","key_t","shift_t","beta_t","gamma_t","g_t" ]
	print [
			(k,v.shape) for k,v in zip(keys,output)
		]






