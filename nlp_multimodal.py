from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from collections import namedtuple
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import json

import nltk.tokenize

import numpy as np
from paths import VALIDATION_FEATURES_DEF as VALIDATION_FEATURES_DEF
from paths import VALIDATION_CAPTIONS_DEF as VALIDATION_CAPTIONS_DEF

from paths import TRAIN_FEATURES_DEF as TRAIN_FEATURES_DEF
from paths import TRAIN_CAPTIONS_DEF as TRAIN_CAPTIONS_DEF
FEATURES_DEF = False
VERBOSE_DEF  = True
VOCAB_FILE 	 = 'vocab.txt'
MIN_WORD_DEFAULT = 4
ImageMetadata = namedtuple("ImageMetadata",
                           ["index","img_url", "captions"])




def _process_caption(caption):
  """Processes a caption string into a list of tokenized words.
  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_caption = []
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  return tokenized_caption


def _load_and_process_metadata(captions_file):
  """Loads image metadata from a JSON file and processes the captions.
  Args:
    captions_file: JSON file containing caption annotations.
  Returns:
    A list of ImageMetadata.
  """
  with open(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the filenames.
  filenames = [x[0] for x in caption_data]

  # Extract the captions. Each image_id is associated with multiple captions.
  url_to_captions = {}
  for captions in caption_data:
    image_id = captions[0]
    caption = captions[1]
    for c in caption:
    	for cc in c:
    		cc = cc.lower()
    url_to_captions.setdefault(image_id,[] )
    url_to_captions[image_id].append(caption)
  print(len(filenames))
  print(len(url_to_captions))
  assert(len(filenames) == len(url_to_captions))
  assert(set([x for x in filenames]) == set(url_to_captions.keys()))
  print("Loaded caption metadata for %d images from %s" %
        (len(url_to_captions), captions_file.split('/')[-1]))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Proccessing captions.")
  image_metadata = []
  num_captions = 0
  # print(url_to_captions)
  for index,url in enumerate(filenames):
    captions = [c for c in url_to_captions[url]]

    image_metadata.append(ImageMetadata(index, url, captions))
    num_captions += len(captions)
  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(filenames), captions_file.split('/')[-1]))
  return image_metadata





def _create_vocab(captions):
  """Creates the vocabulary of word to word_id.
  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.
  Args:
    captions: A list of lists of strings.
  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
  	for cc in c:
    		counter.update(cc)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_words]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with open(FLAGS.vocab_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.vocab_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
    	return self._vocab[word]
    else:
		return self._unk_id

def convert_to_one_hot(train_captions,corpus):
	# x_train = np.zeros((len(train_captions),len(corpus)))
	# for i,captions in enumerate(train_captions):

	raise('TODO')





def initialize():
	'''
	Initialize the data.
	'''
	features = bool(FLAGS.features)
	validation_captions_file = FLAGS.valid_capt
	validation_features_file = FLAGS.valid_feat
	training_features_file   = FLAGS.train_feat
	training_captions_file   = FLAGS.train_capt
	print("Parsing files...")
	train_dataset = _load_and_process_metadata(training_captions_file)
	val_dataset   = _load_and_process_metadata(validation_captions_file)
	# train_features,train_captions,test_features,test_captions = parse_files(training_features_file,
	# 	training_captions_file,validation_features_file,validation_captions_file,features)
	if FLAGS.verbose:
		print("Training Dataset  length:%d"%(len(train_dataset)))
		print("Test     Dataset  length:%d"%(len(val_dataset )))

	train_captions = [c for image in train_dataset for c in image.captions]
	vocab = _create_vocab(train_captions)
	print("Vocabulary length:%d "%(len(vocab._vocab)))
	print("Converting all captions to one-hot enco_ding...")
	# x_train = convert_to_one_hot(train_captions,corpus)
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

	if features ==True:
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
  parser.add_argument("--vocab_file" ,type=str,default=VOCAB_FILE,help="Output vocabulary file of word counts.")
  parser.add_argument("--min_words"   ,type=str,default=MIN_WORD_DEFAULT,help="The minimum number of occurrences of each word "+
  	"in the training set for inclusion in the vocabulary.")
  FLAGS, unparsed = parser.parse_known_args()
  main(None)

