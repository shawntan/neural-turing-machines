import numpy as np
import random

np.random.seed(1234)
random.seed(1234)
def copy(input_size,max_size):
	sequence_length = max_size
	sequence = np.random.binomial(1,0.5,(sequence_length,input_size-1)).astype(np.uint8)
	input_sequence  = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)
	output_sequence = np.zeros((sequence_length*2+1,input_size),dtype=np.float32)
	
	input_sequence[:sequence_length,:-1]  = sequence
	input_sequence[sequence_length,-1] = 1
	output_sequence[sequence_length+1:,:-1] = sequence
	return input_sequence,output_sequence

def repeat_copy(input_size,max_size,num_repeats):
	sequence_length = max_size
	sequence = np.random.binomial(1,0.5,(sequence_length,input_size-1)).astype(np.uint8)
	input_sequence  = np.zeros((sequence_length+1+sequence_length*num_repeats+1,input_size),dtype=np.float32)
	output_sequence = np.zeros((sequence_length+1+sequence_length*num_repeats+1,input_size),dtype=np.float32)
	
	input_sequence[:sequence_length,:-1]  = sequence
	input_sequence[sequence_length,-1] = num_repeats
	output_sequence[sequence_length+1:-1,:-1] = np.tile(sequence, (num_repeats, 1))
	output_sequence[-1,-1] = 1 
	return input_sequence,output_sequence