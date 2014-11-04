import numpy as np
import random
def copy(input_size,max_size):
	sequence_length = random.randrange(max_size) + 1
	sequence = np.random.binomial(1,0.5,(sequence_length,input_size-1)).astype(dtype=np.uint8)
	input_sequence  = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)
	output_sequence = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)
	
	input_sequence[:sequence_length,:-1]  = sequence
	input_sequence[sequence_length,-1] = 1
	output_sequence[sequence_length+1:,:-1] = sequence
	return input_sequence,output_sequence
