
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
from vocabulary import Vocabulary


SEQUENCE_LENGTH_DEFAULT = 32
MAX_SENTENCE_LENGTH_DEFAULT = SEQUENCE_LENGTH_DEFAULT - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE_DEFAULT = 32
CNN_FEATURE_SIZE = 4096
EMBEDDING_SIZE_DEFAULT = 256
EVAL_FREQ_DEFAULT = 100
PRINT_FREQ_DEFAULT = 50
MAX_STEPS_DEFAULT = 10000
MAX_GRAD_NORM = 15
DROPOUT_RATE_DEFAULT = 0.5
TEST_SIZE_DEFAULT = 1000
CAPTION_PRINT_DEFAULT = 10
VALIDATION_SIZE_DEFAULT = 256
VAL_FREQ_DEFAULT = 100
DEFAULT_VOCAB_FILE = 'vocab.txt'
ONE_HOT_DEFAULT = False
LEARNING_RATE_DEFAULT = 1e-3

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
	print("Vocabulary size:%d"%(VOCAB_SIZE))
	# Create model
	# Create model
	l_input_sentence = lasagne.layers.InputLayer((None, FLAGS.seq_length - 1))
	#embedding layer
	l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence, input_size=VOCAB_SIZE,output_size=FLAGS.embedding_size)
	#feature input layer
	l_input_cnn = lasagne.layers.InputLayer((None, CNN_FEATURE_SIZE))
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
	print(l_decoder.output_shape)
	print(l_decoder.input_shape)
	l_out = lasagne.layers.ReshapeLayer(l_decoder, (l_decoder.input_shape[1], FLAGS.seq_length, VOCAB_SIZE))

	return l_input_sentence,l_input_cnn,l_out


def accuracy(logits,sentence_pl,sentences,feature_pl,features):	 
	print('Building predictive function...')

	prediction = lasagne.layers.get_output(logits, 
							{
							 feature_pl : features,
							 sentence_pl: sentences
							}
				)

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
		train_data,test_data,val_data = lstm_utils.get_prepared_merged(max_length=FLAGS.max_sentence,
			validation_size=FLAGS.val_size)
	else:
		print('Loading features/captions')
		train_data,test_data,val_data = lstm_utils.get_merged(max_length = FLAGS.max_sentence,
			validation_size=FLAGS.val_size)


	f = open('results.txt','w+')
	write_flags_to_file(f)

	x_cnn_sym , x_sentence_sym,y_sentence_sym,mask_sym = prepare_placeholders()

	l_in,l_cnn,l_out = inference()


	predictions = accuracy(l_out,l_in,x_sentence_sym,l_cnn,x_cnn_sym)


	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	loss_op = loss(predictions,mask_sym,y_sentence_sym)

	all_grads = T.grad(loss_op, all_params)
	all_grads = [T.clip(g, -5, 5) for g in all_grads]
	all_grads, norm = lasagne.updates.total_norm_constraint(
		 all_grads, MAX_GRAD_NORM, return_norm=True)
	print('Building network optimizer...')

	updates = lasagne.updates.adam(all_grads, all_params, learning_rate=FLAGS.learning_rate)
	print('Testing next batch method.')
	x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(train_data,FLAGS.batch_size)

	print('Building training/test operations...')

	f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
												 [loss_op, norm],
												 updates=updates
												)
	f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], [loss_op,predictions])


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

		if step %FLAGS.eval_freq == 0:
			print('--TODO TESTING')

		if step % FLAGS.val_freq == 0:
			duration = time.time() - start
			try:
				x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(val_data,val_data.features.shape[0])

				loss_val,val_preds = f_val(x_cnn, x_sentence, mask, y_sentence)
				print(val_preds.shape)
				print(y_sentence.shape)

				_print_random_k(val_preds,y_sentence,num_captions=FLAGS.caption_print)

				out =  \
					'==================================================================================\n'+\
					'Step \t%d/%d:\t test_loss =  %8e  (%.4f sec)\n' % (step ,  max_steps , loss_val , duration)+\
					'==================================================================================\n'
				start = time.time()
				print(out)
				f.write(out+"\n")
				f.flush()
			except IndexError:
				print('index error?')
				start = time.time()
				continue 
	f.close()




def _print_random_k(predictions,targets,num_captions=15):
	pred_indices = np.argmax(predictions,axis=2)
	for i in range(num_captions):
		pred = _map_to_sentence(pred_indices[i])
		target = _map_to_sentence(targets[i])
		print('========================================%d/%d image==========================================================='%(i+1,num_captions))
		print('Real caption:\t %s'%(' '.join(target)))
		print('Generated caption:\t %s'%(' '.join(pred)))
		print('===============================================================================================================')
		print('')
		



def write_flags_to_file(f):
	for key, value in vars(FLAGS).items():
	  f.write(key + ' : ' + str(value)+'\n')

def print_flags():
	"""
	Prints all entries in FLAGS variable.
	"""
	for key, value in vars(FLAGS).items():
		print(key + ' : ' + str(value))

def _test_model(dataset,size=1000):
	total_acc = 0.
	test_loss = 0.
	features = dataset.features
	captions = dataset.captions
	test_size = dataset.captions.shape[0]
	test_batches = test_size/size

	chunks = np.array_split(range(test_size),test_batches)

	for i,chunk in enumerate(chunks):
		
		x_sentence = np.zeros((len(captions), FLAGS.seq_length - 1), dtype='int32')
		y_sentence = np.zeros((len(captions), FLAGS.seq_length), dtype='int32')
		mask = np.zeros((len(captions), FLAGS.seq_length), dtype='bool')		
		
		feature_chunk = faetures[chunk]
		caption_chunk = captions[chunk]
		
	
		for j in range(len(features_chunk)):
			x_cnn[j] = feature_chunk[j]
			i = 0
			caption_list = caption_chunk[j].tolist()
			# caption_list = [vocabulary.word_to_id("#START#")] + caption_list + [vocabulary.word_to_id("#END#")] 
			for index,word in enumerate(caption_list):
				mask[j,i] = True
				y_sentence[j, i] = word
				x_sentence[j, i] = word
				i += 1
		test_loss,test_preds = f_val(x_cnn, x_sentence, mask, y_sentence)
		print("running for chunk %d/%d ,test_loss %e"%(i,len(chunks),test_loss))
		test_loss += test_loss

	test_loss /= test_batches

	return test_loss



def prep_batch_for_network(dataset,batch_size):
	
	features,captions,urls = dataset.next_batch(batch_size)
	if FLAGS.one_hot:
		print('apply one-hot encoding')
		num_classes =  len(vocabulary._vocab)
		captions = lstm_utils.dense_to_one_hot(captions, num_classes)

	x_cnn = floatX(np.zeros((len(features), 4096)))
	x_sentence = np.zeros((len(captions), FLAGS.seq_length - 1), dtype='int32')
	y_sentence = np.zeros((len(captions), FLAGS.seq_length), dtype='int32')
	mask = np.zeros((len(captions), FLAGS.seq_length), dtype='bool')
	for j in range(len(features)):
		x_cnn[j] = features[j]
		i = 0
		
		caption_list = captions[j].tolist()

		# print(np.array(caption_list).shape)
		# print(max(caption_list[0]))
		# print(caption_list[0].index(max(caption_list[0])))
		if not FLAGS.one_hot:
			start_tk 	= [start_v]
			end_tk 		= [end_v]
		else:
			start_tk 			= np.zeros((1,num_classes),dtype='int32')
			start_tk[0,start] 	= 1
			end_tk 				= np.zeros((1,num_classes),dtype='int32')
			end_tk[0,end] 		= 1
			start_tk 			= start_tk.tolist()
			end_tk 				= end_tk.tolist()


		# caption_list = start_tk + caption_list + end_tk

		mapped  = _map_to_sentence(caption_list)
		# print(mapped)
		for index,word in enumerate(caption_list):
			mask[j,i] 			= True
			y_sentence[j, i] 	= word
			x_sentence[j, i] 	= word
			i += 1

	return x_cnn, x_sentence, y_sentence, mask


def _map_to_sentence(caption):
	word_list = []
	for word_id in caption:
		if FLAGS.one_hot:
			indices = np.where(np.array(word_id)==1)
			for i,ind in enumerate(indices):
				word_list.append(vocabulary.id_to_word(ind))
		else:
			word_list.append(vocabulary.id_to_word(word_id))
	# print('\n')
	return word_list

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
	parser.add_argument('--test_batch',type=int,default=TEST_SIZE_DEFAULT,
						help='Size of batch for testing')
	parser.add_argument('--caption_print',type=int,default=CAPTION_PRINT_DEFAULT,
						help='captions to print for subjective validation')
	parser.add_argument('--val_size',type=int,default=VALIDATION_SIZE_DEFAULT,
						help='validation size')
	parser.add_argument('--val_freq',type=int,default=VAL_FREQ_DEFAULT,
						help='Frequency of evaluation on validation set')
	parser.add_argument('--vocab_file',type=str,default=DEFAULT_VOCAB_FILE,
						help='Default vocabulary file')
	parser.add_argument('--one_hot',type=str,default=ONE_HOT_DEFAULT,
						help='apply one hot encoding')
	FLAGS, unparsed = parser.parse_known_args()

	vocabulary  = Vocabulary(FLAGS.vocab_file,None,None,flag='load')
	VOCAB_SIZE = len(vocabulary._vocab) 
	start_v = vocabulary.word_to_id("#START#")
	end_v = vocabulary.word_to_id("#END#") 
	print(vocabulary.word_to_id('dressing'))
	main(None)