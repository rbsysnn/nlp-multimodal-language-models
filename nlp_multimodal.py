from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np
VALIDATION_FEATURES_DEF = './datasets/merged_val.npy'
VALIDATION_CAPTIONS_DEF = './datasets/merged_val.json'
TRAIN_FEATURES_DEF      = './datasets/merged_train.npy'
TRAIN_CAPTIONS_DEF      = './datasets/merged_train.json'
FEATURES_DEF            = False
VERBOSE_DEF             = True

def initialize():
	features = bool(FLAGS.features)
	validation_captions_file = FLAGS.valid_capt
	validation_features_file = FLAGS.valid_feat
	training_features_file   = FLAGS.train_feat
	training_captions_file   = FLAGS.train_capt
	print("Parsing files...")
	train_features,train_captions,test_features,test_captions = parse_files(training_features_file,
		training_captions_file,validation_features_file,validation_captions_file,features)
	if FLAGS.verbose:
		print("Training Dataset  length:%d"%(len(train_features)))
		print("Test     Dataset  length:%d"%(len(test_features )))


	print("Creating corpus...")
	corpus = gather_captions(train_captions)
	print("Converting all captions to one-hot encoding...")
	# TODO x_train,x_test = convert_to_one_hot(train_captions,test_captions)
	# if FLAGS.verbose:
		# print_corpus(corpus)


def gather_captions(image_captions):
	corpus = set()
	# print("Images to parse %d"%(len(image_captions)))
	for img_index,img in enumerate(image_captions):
		url = image_captions[img_index][0]
		for i,captions in enumerate(image_captions[img_index][1]):
			for word in captions:
				corpus.add(word)
	return corpus


def print_corpus(corpus):
	print("Corpus length:%d" %(len(corpus)))
	for index,word in enumerate(corpus):
		print(str(word.encode('utf-8')))
		

def _write_flags_to_file(f):
    for key, value in vars(FLAGS).items():
      f.write(key + ' : ' + str(value)+'\n')
def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' \t:\t ' + str(value))


def parse_files(training_features_file = TRAIN_FEATURES_DEF,
	training_captions_file = TRAIN_CAPTIONS_DEF,
	validation_features_file= VALIDATION_FEATURES_DEF,
	validation_captions_file = VALIDATION_CAPTIONS_DEF,features = FEATURES_DEF):

	train_features  = []
	train_captions  = []
	test_features   = []
	test_captions   = []

	if features:
		print("Gathering image features also...")
		train_features = np.load(training_features_file)
		test_features  = np.load(validation_features_file)
	print("Loading captions...")
	
	with open(training_captions_file) as f:
		train_captions = json.load(f)

	with open(validation_captions_file) as f:
		test_captions = json.load(f)

	return train_features,train_captions,test_features,test_captions

def main(_):
	print("Hello")
	print_flags()
	initialize()
	

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--valid_feat' ,type=str,default=VALIDATION_FEATURES_DEF,help='Validation features file')
  parser.add_argument('--valid_capt' ,type=str,default=VALIDATION_CAPTIONS_DEF,help='Validation captions file')
  parser.add_argument('--train_feat' ,type=str,default=TRAIN_FEATURES_DEF     ,  help='Training features file')
  parser.add_argument('--train_capt' ,type=str,default=TRAIN_CAPTIONS_DEF     ,  help='Training captions file')
  parser.add_argument('--features'   ,type=str,default=FEATURES_DEF,help='Parse features (true/false)')
  parser.add_argument('--verbose'    ,type=str,default=VERBOSE_DEF,help='Show further output (true/false)')
  FLAGS, unparsed = parser.parse_known_args()
  main(None)

