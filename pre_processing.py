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
import paths
from vocabulary import Vocabulary

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
    		caption[i] = word
    image_metadata.append(ImageMetadata(index, image_id, caption_list))
    num_captions += len(caption_list)

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

    # Shuffle the ordering of images. Make the randomization repeatable.
    np.random.seed(42)
    np.random.shuffle(images)
    features = np.load(feature_file)

    dataset = []
    # Create a mechanism for monitoring when all threads are finished.
    for image in images:
        for caption in image.captions:
            encoded_caption = []
            for i, word in enumerate(caption):
                # Encode word to index
                encoded_caption.append(vocab.word_to_id(word))
            # Append encoded captions to
            encoded_caption = np.array(encoded_caption)
            input_vec = np.array([features[image.index], encoded_caption])
            print("Appending to dataset image %d\t" % (image.index))
            dataset.append(input_vec)

    dataset = np.array(dataset)
    if name == 'training':
      size = 60000
    else:
      size = 20000
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    dataset = dataset[indices[0:size]]
    # Save processed data

    if not os.path.exists(paths.PROCESSED_FOLDER):
        os.makedirs(paths.PROCESSED_FOLDER)
    feature_file = str(feature_file).split('/')[-1]
    file_to_save = paths.PROCESSED_FOLDER + feature_file

    print("Final dataset size:%s" % (str(dataset.shape)), 'in ', file_to_save)
    np.save(file_to_save, np.array(dataset))

def initialize():
	'''
	Initialize the data.
	'''
	features = bool(FLAGS.features)

	print("----------------------------- Parsing targets -----------------------------")
	train_dataset = _process_metadata(paths.TRAIN_CAPTIONS_DEF)
	val_dataset   = _process_metadata(paths.VALIDATION_CAPTIONS_DEF)

	if FLAGS.verbose:
		print("Training Dataset  length: %d"%(len(train_dataset)))
		print("Test     Dataset  length: %d"%(len(val_dataset )))


	print("--------------------------- Creating vocabulary ---------------------------")
	train_captions = [caption for image in train_dataset for caption in image.captions]
	vocab = _create_vocab(train_captions)

	if FLAGS.verbose:
		print("Vocabulary length:%d "%(len(vocab._vocab)))

	print("---------------------------- Parsing  features ----------------------------")
	_process_dataset('training',train_dataset, vocab, paths.TRAIN_FEATURES_DEF)
	_process_dataset('test',val_dataset, vocab, paths.VALIDATION_FEATURES_DEF)


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' \t:\t ' + str(value))

def main(_):
	print("Welcome to our show and tell version")
	print_flags()
	initialize()


if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--features'   ,type=str,default=FEATURES_DEF,help='Parse features (true/false)')
  parser.add_argument('--verbose'    ,type=str,default=VERBOSE_DEF,help='Show further output (true/false)')
  parser.add_argument("--vocab_file" ,type=str,default=VOCAB_FILE,help="Output vocabulary file of word counts.")
  parser.add_argument("--min_words"  ,type=str,default=MIN_WORD_DEFAULT,help="The minimum number of occurrences of each word "+
  	"in the training set for inclusion in the vocabulary.")
  FLAGS, unparsed = parser.parse_known_args()
  main(None)
