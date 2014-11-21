Neural Turing Machines
======================

Attempt at implementing system described in ["Neural Turing Machines." by Alex Graves, Greg Wayne, and Ivo Danihelka](http://arxiv.org/abs/1410.5401).

This is still very much a work in progress:
- Only the copy task has been implemented.
- Cost does not consistently converge.
- The structure of the controller is merely speculation on my part,
  the paper does not provide the details.

I've documented some of my adventures on my blog, you can check out the articles
on this particular project [here](https://blog.wtf.sg/category/neural-turing-machines/).

There's been some discussion going on in the Reddit thread on /r/machinelearning.
You can check that out [here](https://www.reddit.com/r/MachineLearning/comments/2m9mga/implementation_of_neural_turing_machines/).

## Usage
```
python train_copy.py [output_model_file]
```
