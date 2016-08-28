#!/bin/bash
for i in {0..9}
do
	THEANO_FLAGS=device=gpu0 python -u train_copy.py model.${i}.pkl > ${i}.log
done
