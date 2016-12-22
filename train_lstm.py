
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
import uuid
import datetime
import bleuscore
SEQUENCE_LENGTH_DEFAULT = 32
MAX_SENTENCE_LENGTH_DEFAULT = SEQUENCE_LENGTH_DEFAULT - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE_DEFAULT = 32
CNN_FEATURE_SIZE = 4096
EMBEDDING_SIZE_DEFAULT = 512
EVAL_FREQ_DEFAULT = 5000
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
CHECKPOINT_FREQ_DEFAULT = 2000
MODEL_NAME_DEFAULT = 'tmp_model'
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
	l_out = lasagne.layers.ReshapeLayer(l_decoder, (FLAGS.batch_size, FLAGS.seq_length, VOCAB_SIZE))

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

	start_initial = time.time()
	session_name = './'+FLAGS.name +'/'

	if not os.path.exists(session_name):
		os.makedirs(session_name)

	train_monitor = session_name + 'results.csv'
	validation_monitor = session_name + 'validation_results.csv'
	test_monitor = session_name + 'test_results.csv'
	meta = session_name + 'meta.csv'
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
	train_file = open(train_monitor,'w+')
	train_file.write('Step,Loss,Gradient Norm\n')
	val_file   = open(validation_monitor,'w+')
	val_file.write('Step,Loss,Bleu-1,Bleu-4\n')
	test_file  = open(test_monitor,'w+')
	test_file.write('Step,Loss,Bleu-1,Bleu-4\np')
	meta_info  = open(meta,'w+')
	write_flags_to_file(meta_info)

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

	updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=FLAGS.learning_rate)
	print('Testing next batch method.')
	x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(train_data,FLAGS.batch_size)
	
	f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], [loss_op,predictions])
	file = session_name+'test.csv'
	f = open(file,'w+')
	


	# print('Testing test model method.')
	# test_loss,test_bleu1,test_bleu4 = test_model(test_data,0,f,f_val)

	# print('Testing vali model method.')
	# val_loss,val_bleu1,val_bleu4 = validate_model(val_data,f_val)
	print('Building training/test operations...')
	f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
												 [loss_op, norm],
												 updates=updates
												)
	


	max_steps = FLAGS.max_steps


	for step in range(max_steps):

		start =  time.time()
		x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(train_data,FLAGS.batch_size)
		train_loss, norm = f_train(x_cnn, x_sentence, mask, y_sentence)
		
		
		if step % FLAGS.print_freq == 0:
			duration = time.time() - start
			out =  \
				'==================================================================================\n'+\
				'Step \t%d/%d:\t train_loss =  %8e  \t norm %3.5f (%.4f sec)\n' % (step ,  max_steps , train_loss ,norm , duration)+\
				'==================================================================================\n'
			print(out)
			line = '%s,%s,%s\n'%(str(step),str(train_loss),str(norm))
			train_file.write(line)
			train_file.flush()

		start = time.time()

		if step %FLAGS.check_freq == 0 and step != max_steps:
			
			file = session_name+'checkpoint_'+str(step)+'.csv'
			f = open(file,'w+')
			f.write("id,prediction,target,bleu-1,bleu-4,url\n")
			out =  \
				'==================================================================================\n'+\
				'Saving checkpoint at step %d\t in filename %s\n'%(step,file)+\
				'==================================================================================\n'
			print(out)
			test_loss,test_bleu1,test_bleu4 = test_model(test_data,step,f,f_val)
			duration = time.time() - start
			out =  \
				'==================================================================================================\n'+\
				'Step \t%d/%d:\t test_loss \t=  %2.8f \t bleu1 = %2.5f \t bleu4 %2.5f  (%.4f sec)\n' % (step ,  max_steps , test_loss ,test_bleu1,test_bleu4 , duration)+\
				'==================================================================================================\n'
			start = time.time()

			line = '%s,%s,%s,%s\n'%(str(step),str(test_loss),str(test_bleu1),str(test_bleu4))
			test_file.write(line)
			test_file.flush()
			print(out)

		if step % FLAGS.val_freq == 0 and step != max_steps:

			val_loss,val_bleu1,val_bleu4 = validate_model(val_data,f_val)

			duration = time.time() - start

			out =  \
				'==================================================================================================\n'+\
				'Step \t%d/%d:\t validation_loss \t=  %2.8f \t bleu1 = %2.5f \t bleu4 %2.5f  (%.4f sec)\n' % (step ,  max_steps , val_loss,val_bleu1,val_bleu4, duration)+\
				'==================================================================================================\n'
			print(out)
			line = '%s,%s,%s,%s\n'%(str(step),str(val_loss),str(val_bleu1),str(val_bleu4))
			val_file.write(line)
			val_file.flush()
			start = time.time()


	total_duration = time.time() - start_initial

	train_file.close()
	val_file.close()
	test_file.close()
	meta_info.close()




def _print_random_k(predictions,targets,bleu1,bleu4,num_captions=15):
	pred_indices = np.argmax(predictions,axis=2)
	for i in range(num_captions):
		pred = _map_to_sentence(pred_indices[i])
		target = _map_to_sentence(targets[i])
		print('========================================%d/%d image==========================================================='%(i+1,num_captions))
		print('Real caption:\t%s'%(' '.join(target)))
		print('Generated caption:\t%s'%(' '.join(pred)))
		print('Bleu-1:\t%s\tBleu-4\t%s'%(bleu1[i],bleu4[i]))
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

def test_model(dataset,step,file,f_val,size=32):
	
	total_bleu1 = 0.
	total_bleu4 = 0.
	test_loss = 0.
	features = dataset.features[0:6400]
	captions = dataset.captions[0:6400]
	urls = dataset.urls[0:6400]
	test_size = captions.shape[0]
	test_batches = round(test_size/size)
	chunks = np.array_split(range(test_size),test_batches)


	preds = np.zeros((captions.shape[0]))
	if FLAGS.one_hot:
		print('apply one-hot encoding')
		num_classes =  len(vocabulary._vocab)
		captions = lstm_utils.dense_to_one_hot(captions, num_classes)
	for c,chunk in enumerate(chunks):
				
		feature_chunk = features[chunk]
		caption_chunk = captions[chunk]
		url_chunk 	  = urls[chunk]
		x_cnn = floatX(np.zeros((len(feature_chunk), 4096)))
		x_sentence = np.zeros((len(caption_chunk), FLAGS.seq_length - 1), dtype='int32')
		y_sentence = np.zeros((len(caption_chunk), FLAGS.seq_length), dtype='int32')
		mask = np.zeros((len(caption_chunk), FLAGS.seq_length), dtype='bool')

		#### RESHAPE SENTENCES TO FEED TO THE NETWORK ##############
		for j in range(len(feature_chunk)):
			x_cnn[j] = feature_chunk[j]
			i = 0
			caption_list = caption_chunk[j].tolist()

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

			# caption_list = [vocabulary.word_to_id("#START#")] + caption_list + [vocabulary.word_to_id("#END#")] 

			mapped  = _map_to_sentence(caption_list)
			# print(mapped)
			for index,word in enumerate(caption_list):
				mask[j,i] = True
				y_sentence[j, i] = word
				x_sentence[j, i] = word
				i += 1

		chunk_loss,val_preds = f_val(x_cnn, x_sentence, mask, y_sentence)
		bleu1_chunk = np.zeros((len(caption_chunk),1))
		bleu4_chunk = np.zeros((len(caption_chunk),1))
		test_preds = np.argmax(val_preds,axis=2)

		for k,sentence in enumerate(caption_chunk):
			real_length = len(sentence)
			sentence = _map_to_sentence(sentence)
			pred = _map_to_sentence(test_preds[k])
			pred = pred[0:real_length]
			bleu1_chunk[k],bleu4_chunk[k] = bleuscore.bleu_score(sentence,pred)
			# print('%s,%s,%s,%s,%s,%s\n'%(str(i),str(pred),str(sentence),str(bleu1_chunk[i]),str(bleu4_chunk[i]),str(url_chunk[i])))
			file.write('%s,%s,%s,%s,%s,%s\n'%(str(k*(c)+1),'\"'+str(' '.join(pred))+'\"','\"'+str(' '.join(sentence))+'\"',str(bleu1_chunk[k][0]),str(bleu4_chunk[k][0]),str(url_chunk[k])))
			file.flush()

		total_bleu1 += np.sum(bleu1_chunk)
		total_bleu4 += np.sum(bleu4_chunk)
		test_loss += chunk_loss
		# print("running for chunk %d/%d-%d ,test_loss %2.8f bleu1 %2.5f bleu4 %2.5f"%(c,len(chunks),len(chunk),chunk_loss,np.mean(bleu1_chunk),np.mean(bleu4_chunk)))
	
	total_bleu1 /= len(urls)
	total_bleu4 /= len(urls)
	test_loss   /= len(chunks)

	return test_loss,total_bleu1,total_bleu4





def validate_model(dataset,f_val,size=32):
	

	total_bleu1 = 0.
	total_bleu4 = 0.
	test_loss = 0.
	features = dataset.features
	captions = dataset.captions
	urls     = dataset.urls
	test_size = dataset.captions.shape[0]
	test_batches = test_size/size
	chunks = np.array_split(range(test_size),test_batches)
	preds = np.zeros((captions.shape[0]))
	if FLAGS.one_hot:
		print('apply one-hot encoding')
		num_classes =  len(vocabulary._vocab)
		captions = lstm_utils.dense_to_one_hot(captions, num_classes)
	for c,chunk in enumerate(chunks):
			
		
		feature_chunk = features[chunk]
		caption_chunk = captions[chunk]
		url_chunk     = urls[chunk]
		
		x_cnn = floatX(np.zeros((len(feature_chunk), 4096)))
		x_sentence = np.zeros((len(caption_chunk), FLAGS.seq_length - 1), dtype='int32')
		y_sentence = np.zeros((len(caption_chunk), FLAGS.seq_length), dtype='int32')
		mask = np.zeros((len(caption_chunk), FLAGS.seq_length), dtype='bool')	


		for j in range(len(feature_chunk)):
			x_cnn[j] = feature_chunk[j]
			i = 0
			caption_list = caption_chunk[j].tolist()

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

			# caption_list = [vocabulary.word_to_id("#START#")] + caption_list + [vocabulary.word_to_id("#END#")] 

			mapped  = _map_to_sentence(caption_list)

			for index,word in enumerate(caption_list):
				mask[j,i] = True
				y_sentence[j, i] = word
				x_sentence[j, i] = word
				i += 1
		chunk_loss,val_preds = f_val(x_cnn, x_sentence, mask, y_sentence)


		bleu1_chunk = np.zeros((len(caption_chunk),1))
		bleu4_chunk = np.zeros((len(caption_chunk),1))
		test_preds = np.argmax(val_preds,axis=2)

		chunk_loss,val_preds = f_val(x_cnn, x_sentence, mask, y_sentence)
		bleu1_chunk = np.zeros((len(caption_chunk),1))
		bleu4_chunk = np.zeros((len(caption_chunk),1))
		test_preds = np.argmax(val_preds,axis=2)

		indices = np.arange(caption_chunk.shape[0])
		to_print_reals = caption_chunk[indices[0:FLAGS.caption_print]]
		to_print_vals  = val_preds[indices[0:FLAGS.caption_print]]

	
		for k,sentence in enumerate(caption_chunk):
			real_length = len(sentence)
			sentence = _map_to_sentence(sentence)
			pred = _map_to_sentence(test_preds[k])
			pred = pred[0:real_length]
			bleu1_chunk[k],bleu4_chunk[k] = bleuscore.bleu_score(sentence,pred)
			# file.write('%s,%s,%s,%s,%s,%s\n'%(str(k*(c+1)),'\"'+str(' '.join(pred))+'\"','\"'+str(' '.join(sentence))+'\"',str(bleu1_chunk[k]),str(bleu4_chunk[k]),str(url_chunk[k])))

		total_bleu1 += np.sum(bleu1_chunk)
		total_bleu4 += np.sum(bleu4_chunk)
		test_loss += chunk_loss
		# print("running for chunk %d/%d-%d ,test_loss %2.8f bleu1 %2.5f bleu4 %2.5f"%(c,len(chunks),len(chunk),chunk_loss,np.mean(bleu1_chunk),np.mean(bleu4_chunk)))
	
	total_bleu1 /= len(urls)
	total_bleu4 /= len(urls)
	test_loss   /= len(chunks)
		

	return test_loss,total_bleu1,total_bleu4




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
	parser.add_argument('--check_freq',type=int,default=CHECKPOINT_FREQ_DEFAULT,
						help='test and save results ')
	parser.add_argument('--name',type=str,default=MODEL_NAME_DEFAULT,
						help = 'model name')
	FLAGS, unparsed = parser.parse_known_args()

	vocabulary  = Vocabulary(FLAGS.vocab_file,None,None,flag='load')
	VOCAB_SIZE = len(vocabulary._vocab) 
	start_v = vocabulary.word_to_id("#START#")
	end_v = vocabulary.word_to_id("#END#") 
	print(vocabulary.word_to_id('dressing'))
	main(None)