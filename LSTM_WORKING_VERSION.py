
# coding: utf-8

# In[1]:

import pickle
import random
import numpy as np

import theano
import theano.tensor as T
import lasagne

from collections import Counter
from lasagne.utils import floatX


# In[89]:

num_epochs = 50
num_units_lstm = 100
batch_size = 50
vocab_size = 42


# In[90]:

SEQUENCE_LENGTH = 32
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 50
CNN_FEATURE_SIZE = 4096L
EMBEDDING_SIZE = 256


# In[91]:

def calc_cross_ent(net_output, mask, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (-1, vocab_size))
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
    return cost


# In[94]:

# cnn feature vector
x_cnn_sym = T.matrix()

# sentence encoded as sequence of integer word tokens
x_sentence_sym = T.imatrix()

# mask defines which elements of the sequence should be predicted
mask_sym = T.imatrix()

# ground truth for the RNN output
y_sentence_sym = T.imatrix()    

# Create model
l_input_sentence = lasagne.layers.InputLayer((BATCH_SIZE, SEQUENCE_LENGTH - 1))
 
l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence, input_size=vocab_size,output_size=EMBEDDING_SIZE)

l_input_cnn = lasagne.layers.InputLayer((BATCH_SIZE, CNN_FEATURE_SIZE))
   
l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=EMBEDDING_SIZE, nonlinearity=lasagne.nonlinearities.identity)

l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))

l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])

l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
l_lstm = lasagne.layers.LSTMLayer(l_dropout_input, num_units=EMBEDDING_SIZE, unroll_scan=True, grad_clipping=5.)
   
l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
   
l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, EMBEDDING_SIZE))

l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)

l_out = lasagne.layers.ReshapeLayer(l_decoder, (BATCH_SIZE, SEQUENCE_LENGTH, vocab_size))
   
   
   
   
   #Loss
prediction = lasagne.layers.get_output(l_out, {
               l_input_sentence: x_sentence_sym,
               l_input_cnn: x_cnn_sym
               })
loss = T.mean(calc_cross_ent(prediction, mask_sym, y_sentence_sym))

   #updates
MAX_GRAD_NORM = 15

all_params = lasagne.layers.get_all_params(l_out, trainable=True)

all_grads = T.grad(loss, all_params)
all_grads = [T.clip(g, -5, 5) for g in all_grads]
all_grads, norm = lasagne.updates.total_norm_constraint(
   all_grads, MAX_GRAD_NORM, return_norm=True)

updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.001)

   
#test_prediction = lasagne.layers.get_output(network, deterministic=True)
#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
#test_loss = test_loss.mean()
#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),  dtype=theano.config.floatX)

f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
                         [loss, norm],
                         updates=updates
                        )

f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss)
   
   


# In[95]:

train_data = np.load('name_/datasets/merged_train.npy')


# In[114]:

#Test method

x_cnn= floatX(np.array([x[0] for x in train_data]))
x_sentence = np.zeros((50, SEQUENCE_LENGTH-1), dtype='int32')
y_sentence = np.zeros((50, SEQUENCE_LENGTH), dtype='int32')

mask = np.zeros((50, SEQUENCE_LENGTH), dtype='bool')
for i in range(len(train_data)):
    for j in range(len(train_data[i][1])):
        mask[i,j] = True


# In[120]:

print mask[1]


# In[121]:

for k in range(len(train_data)):
    for l in range(len(train_data[k][1])):
        x_sentence[k][l] = train_data[k][1][l]
        y_sentence[k][l] = train_data[k][1][l]


# In[122]:

print (len(x_sentence[0]))
print (len(y_sentence[0]))
print (len(mask[0]))


# In[123]:

loss_train, norm = f_train(x_cnn, x_sentence, mask, y_sentence)


# In[126]:

print loss_train


# In[ ]:



