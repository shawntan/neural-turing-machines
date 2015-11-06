import numpy as np
import random

np.random.seed(1234)
random.seed(1234)


def copy(batch_size, sequence_length, input_size):
    sequences = np.random.binomial(
            1, 0.5,
            (batch_size, sequence_length, input_size)
        ).astype(np.int8)

    input_sequences = np.zeros(
           (batch_size, sequence_length * 2 + 2, input_size + 2),
            dtype=np.int8
        )

    output_sequences = np.zeros(
            (batch_size, sequence_length * 2 + 2, input_size + 2),
            dtype=np.int8
        )

    input_sequences[:, 1:sequence_length+1, :-2] = sequences
    input_sequences[:, 0, -2] = 1
    input_sequences[:, sequence_length + 1, -1] = 1
    output_sequences[:, sequence_length + 2:, :-2] = sequences
    return input_sequences, output_sequences

def repeat_copy(input_size, max_size, num_repeats):
    sequence_length = max_size
    sequence = np.random.binomial(
        1, 0.5, (sequence_length, input_size - 1)).astype(np.uint8)
    input_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.float32)
    output_sequence = np.zeros(
        (sequence_length + 1 + sequence_length * num_repeats + 1, input_size), dtype=np.float32)

    input_sequence[:sequence_length, :-1] = sequence
    input_sequence[sequence_length, -1] = num_repeats
    output_sequence[sequence_length + 1:-1, :-
                    1] = np.tile(sequence, (num_repeats, 1))
    output_sequence[-1, -1] = 1
    return input_sequence, output_sequence
