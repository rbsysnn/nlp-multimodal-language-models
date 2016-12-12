from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from collections import namedtuple
from collections import Counter
import json, os

import threading
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



def _process_metadata(captions_file):
  """Loads image metadata from a JSON file and processes the captions.
  Args:
    captions_file: JSON file containing caption annotations.
  Returns:
    A list of ImageMetadata.
  """
  with open(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the captions. Each image_id is associated with 5 captions.
  image_metadata = []
  num_captions = 0

  for index, image in enumerate(caption_data):
    image_id = image[0]
    caption_list = image[1]
    for caption in caption_list:
    	for i, word in enumerate(caption):
    		caption[i] = word.lower()
    image_metadata.append(ImageMetadata(index, image_id, caption_list))
    num_captions += len(caption_list)
    if index == 9:
    	break

  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(image_metadata), captions_file.split('/')[-1]))

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
  counter = Counter()
  for caption in captions:
  	for word in caption:
    		counter.update([word])
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [word for word in counter.items() if word[1] >= FLAGS.min_words]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with open(FLAGS.vocab_file, "w") as f:
    f.write("\n".join(["%s %d" % (word, count) for word, count in word_counts]))
  print("Wrote vocabulary file:", FLAGS.vocab_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [item[0] for item in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _process_dataset(name, images, vocab, feature_file):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """

    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.index, image.img_url, [caption])
              for image in images for caption in image.captions]

    print(len(images))


    # Shuffle the ordering of images. Make the randomization repeatable.
    np.random.seed(42)
    np.random.shuffle(images)
    features = np.load(feature_file)

    dataset = []
    # Create a mechanism for monitoring when all threads are finished.
    for image in images:
        for caption in image.captions:
            caption_indices = []
            for i, word in enumerate(caption):
                caption_indices.append(vocab.word_to_id(word))

            ind_caption = np.array(caption_indices)

            input_vec = np.array([features[image.index], ind_caption])
            print("Appending to dataset image %d\t" % (image.index))
            dataset.append(input_vec)


    dataset = np.array(dataset)
    print("Final dataset size:%s" % (str(dataset.shape)))
    file_to_save = 'name_' + feature_file
    print (file_to_save)
    directory = 'name_./datasets/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(file_to_save, np.array(dataset))


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

def initialize():
	'''
	Initialize the data.
	'''
	features = bool(FLAGS.features)
	validation_captions_file = FLAGS.valid_capt
	validation_features_file = FLAGS.valid_feat
	training_features_file   = FLAGS.train_feat
	training_captions_file   = FLAGS.train_capt

	print("----------------------------- Parsing targets -----------------------------")
	train_dataset = _process_metadata(training_captions_file)
	val_dataset   = _process_metadata(validation_captions_file)
	
	if FLAGS.verbose:
		print("Training Dataset  length: %d"%(len(train_dataset)))
		print("Test     Dataset  length: %d"%(len(val_dataset )))


	print("--------------------------- Creating vocabulary ---------------------------")
	train_captions = [caption for image in train_dataset for caption in image.captions]
	vocab = _create_vocab(train_captions)

	if FLAGS.verbose:
		print("Vocabulary length:%d "%(len(vocab._vocab)))

	print("---------------------------- Parsing  features ----------------------------")
	_process_dataset('training',train_dataset, vocab, training_features_file)
	_process_dataset('test',val_dataset, vocab, validation_features_file)


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
	print("Welcome to our show and tell version")
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

