import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U

def build(P,name,input_size,mem_width,mem_size):

	P["W_%s_key"%name]   = U.initial_weights(input_size,mem_width)
	P["b_%s_key"%name]   = U.initial_weights(mem_width)
	P["W_%s_shift"%name] = U.initial_weights(input_size,mem_size)
	P["b_%s_shift"%name] = U.initial_weights(mem_size)

	P["W_%s_beta"%name]  = U.initial_weights(input_size)
	P["b_%s_beta"%name]  = 0.
	P["W_%s_gamma"%name] = U.initial_weights(input_size)
	P["b_%s_gamma"%name] = 0.
	P["W_%s_g"%name]     = U.initial_weights(input_size)
	P["b_%s_g"%name]     = 0.
	
	
	def weight_params(x):
		# key
		key_t = T.dot(x,P["W_%s_key"%name]) + P["b_%s_key"%name]

		# shift
		shift_t = U.vector_softmax(T.dot(x,P["W_%s_shift"%name]) + P["b_%s_shift"%name])

		# scalars
		_beta_t  = T.dot(x,P["W_%s_beta"%name])  + P["b_%s_beta"%name]
		_gamma_t = T.dot(x,P["W_%s_gamma"%name]) + P["b_%s_gamma"%name]

		beta_t  = T.nnet.softplus(_beta_t)
		gamma_t = T.nnet.softplus(_gamma_t) + 1

		g_t     = T.nnet.sigmoid(T.dot(x,P["W_%s_g"%name]) + P["b_%s_g"%name])
		return key_t,beta_t,g_t,shift_t,gamma_t
	return weight_params

