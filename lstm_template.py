"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys


# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)


#tanh, just to save "np."
def tanh(x):
    return np.tanh(x)

# The derivative of the tanh function
def dtanh(x):
    return 1 - x*x


# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# data I/O
data = open('RNNAssignment/data/ijcnlp_dailydialog/test/dialogues_test_edit.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 36
hidden_size = 196  # size of hidden layer of neurons
seq_length = 96  # number of steps to unroll the RNN for
learning_rate = 6e-2 # Learning rate
max_updates = 500000
decay=1e-6
momentum=0.9

concat_size = emb_size + hidden_size

#"""
# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std # input gate
Wo = np.random.randn(hidden_size, concat_size) * std # output gate
Wc = np.random.randn(hidden_size, concat_size) * std # c term

bf = np.ones((hidden_size, 1))*2 # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias
"""
# model parameters
# char embedding parameters
Wex = np.ones([emb_size, vocab_size])*std+0.1 # embedding layer

# LSTM parameters
Wf = np.ones([hidden_size, concat_size]) * std+0.1 # forget gate
Wi = np.ones([hidden_size, concat_size]) * std+0.1 # input gate
Wo = np.ones([hidden_size, concat_size]) * std+0.1 # output gate
Wc = np.ones([hidden_size, concat_size]) * std+0.1 # c term

bf = np.zeros((hidden_size, 1)) # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.ones([vocab_size, hidden_size])*0.1 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias
"""

##debug fill params
#all_params = (Wex, Wf, Wi, Wo, Wc, bf, bi, bo, bc, Why, by)
#for param in all_params:
#    for i in range(len(param.flat)):
#        if param.flat[i] is not 0:
#            param.flat[i] = i/len(param.flat)


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    hprev, cprev = memory

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    xs, hs, cs, wes, zs, os, f_gate, i_gate, c_cand, o_gate, ps, ys = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t-1], wes[t]))
        #print ("shape of zs[t]: " + str(zs[t].shape))
        #print ("shape of hs[t-1]: " + str(hs[t-1].shape))
        #print ("shape of wes[t]: " + str(wes[t].shape))

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        # f_gate = sigmoid (Wf \cdot [h X] + bf)
        f_gate[t] = sigmoid(np.dot(Wf, zs[t]) + bf)

        # compute the input gate
        # i_gate = sigmoid (Wi \cdot [h X] + bi)
        i_gate[t] = sigmoid(np.dot(Wi, zs[t]) + bi)

        # compute the candidate memory
        # \hat{c} = tanh (Wc \cdot [h X] + bc])
        c_cand[t] = tanh(np.dot(Wc, zs[t]) + bc)


        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_new = f_gate * prev_c + i_gate * \hat{c}
        cs[t] = f_gate[t] * cs[t-1] + i_gate[t] * c_cand[t]
        #print ("shape of cs[t] "  + str(cs[t].shape))

        # output gate
        # o_gate = sigmoid (Wo \cdot [h X] + bo)
        o_gate[t] = sigmoid (np.dot(Wo, zs[t])+bo)
        #print ("shape of o_gate : "  + str(o_gate.shape))

        # new hidden state for the LSTM
        hs[t] = o_gate[t] * tanh(cs[t])
        #print ("shape of hs[t] : "  + str(hs[t].shape))
        
        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        # o = Why \cdot h + by
        os[t] = np.dot(Why, hs[t]) + by

        # softmax for probabilities for next chars
        # p = softmax(o)
        ps[t] = softmax(os[t])
        

        # cross-entropy loss       
        # cross entropy loss at time t:
        # create an one hot vector for the label y
        ys[t] = np.zeros((vocab_size, 1))
        ys[t][targets[t]] = 1


        
        # and then cross-entropy (see the elman-rnn file for the hint)
        loss_t = np.sum(-np.log(ps[t])*ys[t])
        loss += loss_t


    # define your activations
    activations = (xs, hs, cs, wes, zs, ps, ys, f_gate, i_gate, c_cand, o_gate)
    memory = (hs[len(inputs)-1], cs[len(inputs)-1])

    return loss, activations, memory


def backward(activations, clipping=False):
    """
    during the backward pass we follow the track of the forward pass
    the activations are needed so that we can avoid unnecessary re-computation
    """

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    xs, hs, cs, wes, zs, ps, ys, f_gate, i_gate, c_cand, o_gate = activations

    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])
    #print ("shape of dhnext : "  + str(dhnext.shape))

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):

        # IMPLEMENT YOUR BACKPROP HERE
        # skipping over cross entropy and softmax in backwards pass
        # dL/do becomes ps - ys 
        do = ps[t] - ys[t]

        #o = Why*h_t+by
        dWhy += np.dot(do, hs[t].T)
        dby += do

        #dh[t+1] flows in from future cell
        dh = np.dot(Why.T,do) + dhnext

        # h[t]= o_gate * tanh(c_new)
        #o_gate = sigmoid(o_presig)


        # https://kitchingroup.cheme.cmu.edu/blog/2013/03/12/Potential-gotchas-in-linear-algebra-in-numpy/

        # Numpy has some gotcha features for linear algebra purists. The first is that a 1d array is neither a row, 
        # nor a column vector. That is, a = a.T if a is a 1d array. 
        # That means you can take the dot product of a with itself, without transposing the second argument. 
        # This would not be allowed in Matlab.
        
        #a = np.array([0, 1, 2])
        #print a.shape
        #print a
        #print a.T

        #print
        #print np.dot(a, a)
        #print np.dot(a, a.T)

        #>>> >>> (3L,)
        #[0 1 2]
        #[0 1 2]
        #>>>
        #5
        #5

        # Compare the previous behavior with this 2d array. In this case, you cannot take the dot product of 
        # b with itself, because the dimensions are incompatible. You must transpose the second argument to 
        # make it dimensionally consistent. Also, the result of the dot product is not a simple scalar, but a 1 × 1 array.




        #print ("cs[t]: " + str(cs[t]))
        #print ("dh: " + str(dh))
        #print (cs[t]*dh)
        do_gate = tanh(cs[t])*dh
        do_presig = dsigmoid(o_gate[t]) * do_gate #dsigmoid(o_presig?)
        dWo += np.dot(do_presig, zs[t].T) 
        dbo += do_presig
        #print ("shape of do_gate "  + str(do_gate.shape))
        #print ("shape of do_presig "  + str(do_presig.shape))
        #print ("shape of Wo "  + str(Wo.shape))

        # c_new = c[t-1]*f_gate + i_gate*c_cand
        # future cell state dcnext comes in from future cell 
        dc = dh*o_gate[t]*dtanh(tanh(cs[t])) + dcnext #or ...(dtanh(tanh(cs[t])))
        #print ("shape of dc "  + str(dc.shape))

        # c_new = c[t-1]*f_gate + i_gate*c_cand
        dc_cand = i_gate[t]*dc
        dc_pretanh = dtanh(c_cand[t])*dc_cand #dtanh(c_pretanh) ?
        #print ("shape of dc_pretanh "  + str(dc_pretanh.shape))

        dWc += np.dot(dc_pretanh, zs[t].T) 
        dbc += dc_pretanh
        #print ("shape of dWc "  + str(dWc.shape))
        #print ("shape of dbc "  + str(dbc.shape))

        # c_new = c[t-1]*f_gate + i_gate*c_cand
        di_gate = c_cand[t]*dc
        di_presig = dsigmoid(i_gate[t])*di_gate #dsigmoid(i_presig) ?
        dWi += np.dot(di_presig, zs[t].T) 
        dbi += di_presig
        #print ("shape of di_presig "  + str(di_presig.shape))
        #print ("shape of dWi "  + str(dWi.shape))


        # c_new = c[t-1]*f_gate + i_gate*c_cand
        df_gate = cs[t-1]*dc
        df_presig = dsigmoid(f_gate[t])*df_gate #dsigmoid(f_presig) ?
        dWf += np.dot(df_presig, zs[t].T) 
        dbf += df_presig
        #print ("shape of df_presig "  + str(df_presig.shape))
        #print ("shape of dWf "  + str(dWf.shape))

        #sum up gradients from different paths of partial derivatives
        dz = np.dot(Wf.T, df_presig) +  np.dot(Wi.T, di_presig) + np.dot(Wc.T, dc_pretanh) + np.dot(Wo.T, do_presig)
        #print ("shape of dz "  + str(dz.shape))
        #print (" dz "  + str(dz))

        #print ("shape of wes[t]: " + str(wes[t].shape))
        #print ("len of wes[t]: " + str(len(wes[t])))
        #gradients dEmb wrt inputs is in last rows of z
        #print ("shape of dz[-len(wes[t]):,:]: " + str(dz[-len(wes[t]):,:].shape))
        #print ("dz[-len(wes[t]):,:]: " + str(dz[-len(wes[t]):,:]))

        dWex += np.dot(dz[-len(wes[t]):,:], xs[t].T)
        #print ("shape of dWex: " + str(dWex.shape))
        #print ("dWex: " + str(dWex))

        #gradients dhnext wrt h[t-1] is in the first rows of z
        dhnext = dz[:len(hs[t-1]),:]
        #print ("shape of dhnext: " + str(dhnext.shape))
        #print ("dhnext: " + str(dhnext))

        dcnext = f_gate[t]*dc


    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    generated_chars = []

    for t in range(n):
        # IMPLEMENT THE FORWARD FUNCTION ONE MORE TIME HERE
        # BUT YOU DON"T NEED TO STORE THE ACTIVATIONS
        wes = np.dot(Wex, x)
        z = np.row_stack((h, wes))

        f_gate = sigmoid(np.dot(Wf, z) + bf)
        i_gate = sigmoid(np.dot(Wi, z) + bf)
        c_cand = np.tanh(np.dot(Wc, z) + bc)
        c = f_gate * c + i_gate * c_cand
        o_gate = sigmoid (np.dot(Wo, z)+bo)
        h = o_gate * np.tanh(c)
        o = np.dot(Why, h) + by
        p = softmax(o)
        
        # the the distribution, we randomly generate samples:
        ix = np.random.multinomial(1, p.ravel())
        x = np.zeros((vocab_size, 1))

        for j in range(len(ix)):
            if ix[j] == 1:
                index = j
        #### DEBUGG ####
        #index = 15
        #### DEBUGG ####
        x[index] = 1
        generated_chars.append(index)



    return generated_chars

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            cprev = np.zeros((hidden_size,1))
            p = 0 # go from start of data
        
        ###### DEBUGG ########
        #inputs = [42,14]
        #targets = [14,26]
        ###### DEBUGG ########
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 1000 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print ('----\n %s \n----' % (txt, ))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 1000 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                    [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                    [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            
            #Adagrad
            mem += dparam * dparam
            #RMSProp 
            #mem *= momentum
            #mem += (1-momentum)*dparam*dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        p += seq_length # move data pointer
        n += 1 # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 0.01

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                   [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                   ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert(weight.shape == grad.shape), str_

        print(name)
        for i in range(weight.size):
      
            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / ( 2 * delta )

            # compare the relative error between analytical and numerical gradients
            rel_error = abs(grad_analytic - grad_numerical) / (abs(grad_numerical + grad_analytic)+1e-30)

            if rel_error > 0.1:
                print ("WARNING, num: " +  str(grad_numerical) + ", analytic: " + str(grad_analytic) + " ==> rel. err: " + str(rel_error))
