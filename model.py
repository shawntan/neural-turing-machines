import theano
import theano.tensor as T
import numpy         as np
from theano.printing import Print
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from theano_toolkit import hinton
import controller
import head
import scipy

def cosine_sim(k,M):
	k_unit = k / T.sqrt(T.sum(k**2))
	k_unit = T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
	k_unit.name = "k_unit"
	M_lengths = T.patternbroadcast(T.sqrt(T.sum(M**2,axis=1)).reshape((M.shape[0],1)),(False,True))
	M_unit = M / M_lengths
	M_unit.name = "M_unit"
#	M_unit = Print("M_unit")(M_unit)
	return T.sum(k_unit * M_unit,axis=1)

def build_step(P,controller,controller_size,mem_size,mem_width,similarity=cosine_sim):
	circ_convolve = scipy.linalg.circulant(np.arange(mem_size)).T
	P.memory_init = 0.1 * np.random.randn(mem_size,mem_width)

	P.read_weight_init  = 0.1 * np.random.randn(mem_size)
	P.add_weight_init   = 0.1 * np.random.randn(mem_size)
	P.erase_weight_init = 0.1 * np.random.randn(mem_size)
	
	P.W_erase_head = U.initial_weights(controller_size,mem_width)
	P.b_erase_head = U.initial_weights(mem_width)
	P.W_add_head   = U.initial_weights(controller_size,mem_width)
	P.b_add_head   = U.initial_weights(mem_width)

	memory_init   = P.memory_init
	read_weight_init  = U.vector_softmax(P.read_weight_init)
	erase_weight_init = U.vector_softmax(P.erase_weight_init)
	add_weight_init   = U.vector_softmax(P.add_weight_init)



	heads = { h:head.build(P,h,controller_size,mem_width,mem_size)
				for h in ["read","erase","add"] }
	
	def build_memory_curr(M_prev,erase_head,erase_weight,add_head,add_weight):
		erase_weight = erase_weight.dimshuffle((0,'x'))
		add_weight   = add_weight.dimshuffle((0,'x'))

		erase_head = erase_head.dimshuffle(('x',0))
		add_head   = add_head.dimshuffle(('x',0))

		M_erased = M_prev   * (1 - (erase_weight * erase_head))
		M_curr   = M_erased +      (add_weight   * add_head)
		return M_curr
	
	def build_read(M_curr,weight_curr):
		return T.dot(weight_curr, M_curr)

	def build_weight_curr(weight_prev,M_curr,head,input_curr):
		"""
		This function is best described by Figure 2 in the paper.
		"""
		key,beta,g,shift,gamma = head(input_curr)
		# 3.3.1 Focusing b Content
		weight_c = U.vector_softmax(beta * similarity(key,M_curr))

		# 3.3.2 Focusing by Location
		weight_g = g * weight_c + (1 - g) * weight_prev
		weight_shifted = T.dot(weight_g,shift[circ_convolve])
		weight_sharp   = weight_shifted ** gamma
		weight_curr    = weight_sharp / T.sum(weight_sharp)
		return weight_curr
	
	def step(input_curr,M_prev,
			read_weight_prev,
			erase_weight_prev,
			add_weight_prev):
		#print read_prev.type
		
		read_prev = build_read(M_prev,read_weight_prev)
		output,controller_hidden = controller(input_curr,read_prev)

		read_weight  = build_weight_curr(read_weight_prev,  M_prev, heads["read"],  controller_hidden)
		erase_weight = build_weight_curr(erase_weight_prev, M_prev, heads["erase"], controller_hidden)
		add_weight   = build_weight_curr(add_weight_prev,   M_prev, heads["add"],   controller_hidden)

		erase_head = T.nnet.sigmoid(T.dot(controller_hidden,P.W_erase_head) + P.b_erase_head)
		add_head   = T.dot(controller_hidden,P.W_add_head) + P.b_add_head

		M_curr = build_memory_curr(M_prev,erase_head,erase_weight,add_head,add_weight)
		
		
		#print [i.type for i in [erase_curr,add_curr,key_curr,shift_curr,beta_curr,gamma_curr,g_curr,output]]
		#print weight_curr.type
		return M_curr,read_weight,erase_weight,add_weight,output
	return step,[memory_init,read_weight_init,erase_weight_init,add_weight_init,None]

def build(P,mem_size,mem_width,controller_size,ctrl):
	step,outputs_info = build_step(P,ctrl,controller_size,mem_size,mem_width)
	def predict(input_sequence):
		outputs,_ = theano.scan(
				step,
				sequences    = [input_sequence],
				outputs_info = outputs_info
			)
		return outputs
	
	return predict


