# LSTM-Task 

This is the LSTM  assignment of the Deep Learning and Neural Network course of  ISL at KIT, based on templates by Ngoc Quan Pham.

The implementation of the character based language model with Vanilla RNN and LSTM is included.

# Requirement.
Python 3.7 and Numpy are the only requirement.

First, ensure that you are in the same directory with the python files and the "data" directory. 

For the Vanilla RNN and LSTM  you can run two things:

- Training it to see the loss function and the samples being generated every 1000 steps. You can manually change the hyperparameters to play around with the code a little bit.

python elman_rnn.py train
python lstm.py train

- Check the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple:

+ We need to know how to verify the correctness of the back-prop implementation.
+ In order to do that we rely on comparison with the gradients computed using numerical differentiation
+ For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta)
+ The difference between two forward passes gives us the gradient for that weight
+ (maybe the code will be self-explanationable)

python elman_rnn.py gradcheck
python lstm.py gradcheck
