#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tasks
import sys
import run_model

def plot(seq_length):
    i,o = tasks.copy(8,seq_length)
    weights,outputs = do_task(i)
    plt.figure(1,figsize=(20,7))
    plt.subplot(311)
    plt.imshow(i.T,interpolation='nearest')
    plt.subplot(312)
    plt.imshow(o.T,interpolation='nearest')
    plt.subplot(313)
    plt.imshow(outputs.T,interpolation='nearest')
    plt.show()


def plot_weights(seq_length):
    i,o = tasks.copy(8,seq_length)
    weights,outputs = do_task(i)
    plt.figure(1,figsize=(20,20))
    plt.imshow(weights.T[100:123],interpolation='nearest',cmap=cm.gray)
    plt.show() 

if __name__ == "__main__":
      P,do_task = run_model.make_model()
      P.load('l2_low_learning_rate.mdl')
      plot(130)
      #plot_weights(8)