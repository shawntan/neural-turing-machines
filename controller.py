import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U

import controller
import model
from collections import namedtuple
#from theano_toolkit.parameters import Parameters

def build(P,input_size,output_size,mem_size,mem_width,layer_sizes):
	"""
	Create controller function for use during scan op
	"""

	P.W_input_hidden = U.initial_weights(input_size,layer_sizes[0])
	P.W_read_hidden  = U.initial_weights(mem_width, layer_sizes[0])
	P.b_hidden_0 = 0. * U.initial_weights(layer_sizes[0])
	hidden_weights = []
	for i in xrange(len(layer_sizes)-1):
		P["W_hidden_%d"%(i+1)] = U.initial_weights(layer_sizes[i],layer_sizes[i+1])
		P["b_hidden_%d"%(i+1)] = 0. * U.initial_weights(layer_sizes[i+1])
		hidden_weights.append((P["W_hidden_%d"%(i+1)],P["b_hidden_%d"%(i+1)]))

	P.W_hidden_output = U.initial_weights(layer_sizes[-1],output_size)
	P.b_output = 0. * U.initial_weights(output_size)

	heads = ["read","erase","add"]

	for h in heads:
		P["W_hidden_%s_key"%h]   = U.initial_weights(layer_sizes[-1],mem_width)
		P["W_hidden_%s_shift"%h] = U.initial_weights(layer_sizes[-1],mem_size)
		P["b_%s_key"%h]   = U.initial_weights(mem_width)
		P["b_%s_shift"%h] = U.initial_weights(mem_size)

		P["W_hidden_%s_beta"%h]  = U.initial_weights(layer_sizes[-1])
		P["W_hidden_%s_gamma"%h] = U.initial_weights(layer_sizes[-1])
		P["W_hidden_%s_g"%h]     = U.initial_weights(layer_sizes[-1])
		P["b_%s_beta"%h]  = 0.
		P["b_%s_gamma"%h] = 0.
		P["b_%s_g"%h]     = 0.

		if h != "read":
			P["W_hidden_%s"%h] = U.initial_weights(layer_sizes[-1],mem_width)
			P["b_%s"%h]        = 0. * U.initial_weights(mem_width)


		HeadParams = namedtuple('HeadParams','key shift beta gamma g head')

	def controller(input_t,read_t):
#		print "input_t",input_t.type
		prev_layer = hidden_0 = T.nnet.sigmoid(
				T.dot(input_t,P.W_input_hidden) +\
				T.dot(read_t,P.W_read_hidden) +\
				P.b_hidden_0
			)

#		print "input",read_t.type,input_t.type
#		print "weights",P.W_input_hidden.type,P.W_read_hidden.type,P.b_hidden_0.type
#		print "layer", hidden_0.type
		for W,b in hidden_weights:
			prev_layer = T.nnet.sigmoid(T.dot(prev_layer,W) + b)
		fin_hidden = prev_layer
		
		output_t = T.nnet.sigmoid(T.dot(fin_hidden,P.W_hidden_output) + P.b_output)
		
		head_params = []
		for h in heads:
			
			# key
			key_t = T.dot(fin_hidden,P["W_hidden_%s_key"%h]) + P["b_%s_key"%h]

			# shift
			shift_t = U.vector_softmax(T.dot(fin_hidden,P["W_hidden_%s_shift"%h]) + P["b_%s_shift"%h])
			shift_t.name = "%s_shift_t"%h

			# scalars
			_beta_t  = T.dot(fin_hidden,P["W_hidden_%s_beta"%h])  + P["b_%s_beta"%h]
			_gamma_t = T.dot(fin_hidden,P["W_hidden_%s_gamma"%h]) + P["b_%s_gamma"%h]

			beta_t  = (_beta_t > 0) * _beta_t
			gamma_t = (_gamma_t > 0 ) * _gamma_t + 1

			g_t      = T.nnet.sigmoid(T.dot(fin_hidden,P["W_hidden_%s_g"%h]) + P["b_%s_g"%h])

			if h == "erase":
				head_t = T.nnet.sigmoid(T.dot(fin_hidden,P["W_hidden_%s"%h]) + P["b_%s"%h])
			elif h == "add":
				head_t = T.dot(fin_hidden,P["W_hidden_%s"%h]) + P["b_%s"%h]
			elif h == "read":
				head_t = None

			head_params.append(
					HeadParams(
						key   = key_t,
						head  = head_t,
						shift = shift_t,
						beta  = beta_t,
						gamma = gamma_t,
						g = g_t
					))
		return output_t,head_params

	return controller

