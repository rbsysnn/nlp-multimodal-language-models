
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
import os
import argparse
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


VOCAB_SIZE = lstm_utils.get_num_classes('vocab.txt')



SEQUENCE_LENGTH_DEFAULT = 32
MAX_SENTENCE_LENGTH_DEFAULT = SEQUENCE_LENGTH_DEFAULT - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE_DEFAULT = 128
CNN_FEATURE_SIZE = 4096
EMBEDDING_SIZE_DEFAULT = 256
EVAL_FREQ_DEFAULT = 200
PRINT_FREQ_DEFAULT = 10
MAX_STEPS_DEFAULT = 10000
MAX_GRAD_NORM = 15
DROPOUT_RATE_DEFAULT = 0.5

LEARNING_RATE_DEFAULT = 1e-4

def calc_cross_ent(net_output, mask, targets):
		# Helper function to calculate the cross entropy error
		preds = T.reshape(net_output, (-1, VOCAB_SIZE))
		targets = T.flatten(targets)
		cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
		return cost


def prepare_placeholders():
	print('Setting placeholders')

	x_cnn_sym = T.matrix()

	# sentence encoded as sequence of integer word tokens
	x_sentence_sym = T.imatrix()

	# mask defines which elements of the sequence should be predicted
	mask_sym = T.imatrix()

	# ground truth for the RNN output
	y_sentence_sym = T.imatrix()  

	return x_cnn_sym,x_sentence_sym,y_sentence_sym,mask_sym


def inference():
	print('Building model...')

	# Create model
	l_input_sentence = lasagne.layers.InputLayer((FLAGS.batch_size, FLAGS.seq_length - 1))
	#embedding layer
	l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence, input_size=VOCAB_SIZE,output_size=FLAGS.embedding_size)
	#feature input layer
	l_input_cnn = lasagne.layers.InputLayer((FLAGS.batch_size, CNN_FEATURE_SIZE))
	#fully connected embedding of features layer
	l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=FLAGS.embedding_size, nonlinearity=lasagne.nonlinearities.identity)
	#flatten
	l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))
	#concatenate word and feature embeddings
	l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])
	#dropout layer
	l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=FLAGS.dropout)
	#lstm module
	l_lstm = lasagne.layers.LSTMLayer(l_dropout_input, num_units=FLAGS.embedding_size, unroll_scan=True, grad_clipping=5.)
	#dropout layer
	l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=FLAGS.dropout)
	#flatten lstm
	l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, FLAGS.embedding_size))
	#softmax layer
	l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=VOCAB_SIZE, nonlinearity=lasagne.nonlinearities.softmax)
	#reshape to vocab_size
	l_out = lasagne.layers.ReshapeLayer(l_decoder, (FLAGS.batch_size, FLAGS.seq_length, VOCAB_SIZE))

	return l_input_sentence,l_input_cnn,l_out


def prediction(logits,sentence_pl,sentences,feature_pl,features):	 
	print('Building predictive function...')
	prediction = lasagne.layers.get_output(logits, {
							 sentence_pl: sentences,
							 feature_pl : features
							 })
	return prediction


def loss(prediction,mask,ground_truth):

	loss = T.mean(calc_cross_ent(prediction, mask, ground_truth))

	return loss


	 #updates


def train():

	print('Preparing data...')
	file = './datasets/processed/'+str(FLAGS.max_sentence)
	print(file)
	if os.path.exists(file):
		print('Loading preprocessed features/captions')
		train_data,test_data,_ = lstm_utils.get_prepared_merged(one_hot=False,max_length=FLAGS.max_sentence)
	else:
		print('Loading features/captions')
		train_data,test_data,_ = lstm_utils.get_merged(one_hot=False,max_length = FLAGS.max_sentence)


	f = open('results.txt','w+')
	write_flags_to_file(f)

	x_cnn_sym , x_sentence_sym,y_sentence_sym,mask_sym = prepare_placeholders()

	l_in,l_cnn,l_out = inference()

	predictions = prediction(l_out,l_in,x_sentence_sym,l_cnn,x_cnn_sym)

	loss_op = loss(predictions,mask_sym,y_sentence_sym)

	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	all_grads = T.grad(loss_op, all_params)
	all_grads = [T.clip(g, -5, 5) for g in all_grads]
	all_grads, norm = lasagne.updates.total_norm_constraint(
		 all_grads, MAX_GRAD_NORM, return_norm=True)
	print('Building network optimizer...')

	updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.001)

	print('Building training/test operations...')

	f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
												 [loss_op, norm],
												 updates=updates
												)
	f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss_op)

	print("Starting training...")
	max_steps = FLAGS.max_steps
	for step in range(max_steps):
		start =  time.time()
		x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(train_data,FLAGS.batch_size)

		train_loss, norm = f_train(x_cnn, x_sentence, mask, y_sentence)
		
		duration = time.time() - start
		if step % FLAGS.print_freq == 0:
			out =  \
				'==================================================================================\n'+\
				'Step \t%d/%d:\t train_loss =  %8e  \t norm %3.5f (%.4f sec)\n' % (step ,  max_steps , train_loss ,norm , duration)+\
				'==================================================================================\n'
			print(out)
			f.write(out+"\n")
			f.flush()
		start = time.time()

		if step % FLAGS.eval_freq == 0:
			duration = time.time() - start
			try:
				x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(test_data,FLAGS.batch_size)
				loss_val = f_val(x_cnn, x_sentence, mask, y_sentence)
				out =  \
					'==================================================================================\n'+\
					'Step \t%d/%d:\t test_loss =  %8e  (%.4f sec)\n' % (step ,  max_steps , loss_val , duration)+\
					'==================================================================================\n'
				start = time.time()
				print(out)
				f.write(out+"\n")
				f.flush()
			except IndexError:
				start = time.time()
				continue 
	f.close()



def write_flags_to_file(f):
	for key, value in vars(FLAGS).items():
	  f.write(key + ' : ' + str(value)+'\n')

def print_flags():
	"""
	Prints all entries in FLAGS variable.
	"""
	for key, value in vars(FLAGS).items():
		print(key + ' : ' + str(value))

def prep_batch_for_network(dataset,batch_size):
	features,captions = dataset.next_batch(batch_size)

	x_cnn = floatX(np.zeros((len(features), 4096)))
	x_sentence = np.zeros((len(captions), FLAGS.seq_length - 1), dtype='int32')
	y_sentence = np.zeros((len(captions), FLAGS.seq_length), dtype='int32')
	mask = np.zeros((len(captions), FLAGS.seq_length), dtype='bool')
	for j in range(len(features)):

		x_cnn[j] = features[j]
		i = 0
		for word in captions[j]:
			mask[j,i] = True
			y_sentence[j, i] = word
			x_sentence[j, i] = word
			i += 1
	return x_cnn, x_sentence, y_sentence, mask



def main(_):
	print_flags()
	train()

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
						help='Learning rate')
	parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
						help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
						help='Batch size to run trainer.')
	parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
						help='Frequency of evaluation on the train set')
	parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
						help='Frequency of evaluation on the test set')
	parser.add_argument('--seq_length',type=int,default=SEQUENCE_LENGTH_DEFAULT,
						help='Sequence for lstm to accept')
	parser.add_argument('--max_sentence',type=int,default=MAX_SENTENCE_LENGTH_DEFAULT,
						help='Maximum length of sentence to predict')
	parser.add_argument('--embedding_size',type=int,default=EMBEDDING_SIZE_DEFAULT,
						help='Default embedding size')
	parser.add_argument('--dropout',type=float,default=0.5,
						help='Dropout rate')

	FLAGS, unparsed = parser.parse_known_args()
	main(None)