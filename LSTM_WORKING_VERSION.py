
# coding: utf-8

# In[1]:

import pickle
import random
import numpy as np

import theano
import theano.tensor as T
import lasagne
from theano import function, config, shared, sandbox

from collections import Counter
from lasagne.utils import floatX
import lstm_utils
import time
# # In[2]:
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 100

# rng = np.random.RandomState(22)
# x = shared(np.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
# 	r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
# 	print('Used the cpu')
# else:
# 	print('Used the gpu')


num_epochs = 50
num_units_lstm = 100
batch_size = 50
vocab_size = lstm_utils.get_num_classes('vocab.txt') + 1


# In[58]:

SEQUENCE_LENGTH = 48
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 50
CNN_FEATURE_SIZE = 4096L
EMBEDDING_SIZE = 256


# In[59]:

def calc_cross_ent(net_output, mask, targets):
		# Helper function to calculate the cross entropy error
		preds = T.reshape(net_output, (-1, vocab_size))
		targets = T.flatten(targets)
		cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
		return cost


# In[60]:
print('Setting placeholders')
# cnn feature vector
x_cnn_sym = T.matrix()

# sentence encoded as sequence of integer word tokens
x_sentence_sym = T.imatrix()

# mask defines which elements of the sequence should be predicted
mask_sym = T.imatrix()

# ground truth for the RNN output
y_sentence_sym = T.imatrix()    
print('Building model...')
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
	 
	 
print('Building predictive function...')
	 
	 
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


print('Building network optimizer...')
updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.001)

	 
#test_prediction = lasagne.layers.get_output(network, deterministic=True)
#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
#test_loss = test_loss.mean()
#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),  dtype=theano.config.floatX)
f = open('results.txt','w+')
print('Building training/test operations...')
f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
												 [loss, norm],
												 updates=updates
												)

f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss)
	 


# In[61]:

# train_data = np.load('datasets/processed/merged_train.npy')


# In[62]:

#Test method
print('Preparing data...')
train_data,test_data,_ = lstm_utils.get_merged(one_hot=False)



def prep_batch_for_network(dataset,batch_size):
	features,captions = dataset.next_batch(batch_size)

	x_cnn = floatX(np.zeros((len(features), 4096)))
	x_sentence = np.zeros((len(captions), SEQUENCE_LENGTH - 1), dtype='int32')
	y_sentence = np.zeros((len(captions), SEQUENCE_LENGTH), dtype='int32')
	mask = np.zeros((len(captions), SEQUENCE_LENGTH), dtype='bool')
	for j in range(len(features)):

		x_cnn[j] = features[j]
		i = 0
		for word in captions[j]:
			mask[j,i] = True
			y_sentence[j, i] = word
			x_sentence[j, i] = word
			i += 1
	return x_cnn, x_sentence, y_sentence, mask


max_steps = 20000
for step in range(max_steps):
		
		start =  time.time()
		x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(train_data,BATCH_SIZE)

		train_loss, norm = f_train(x_cnn, x_sentence, mask, y_sentence)
		duration = start - time.time()
		
		out =  \
			'==================================================================================\n'+\
			'Step \t%d/%d:\t train_loss =  %8e  \t norm %3.5f (%.4f sec)\n' % (step ,  max_steps , train_loss ,norm , duration)+\
			'==================================================================================\n'

		f.write(out+"\n")
		f.flush()
		start = time.time()
		if not step % 250:
			duration = start - time.time()
			try:
				x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(test_data,BATCH_SIZE)
				loss_val = f_val(x_cnn, x_sentence, mask, y_sentence)
				out =  \
					'==================================================================================\n'+\
					'Step \t%d/%d:\t train_loss =  %8e  (%.4f sec)\n' % (step ,  max_steps , loss_val , duration)+\
					'==================================================================================\n'
				start = time.time()
				f.write(out+"\n")
				f.flush()
			except IndexError:
				start = time.time()
				continue 

f.close()

