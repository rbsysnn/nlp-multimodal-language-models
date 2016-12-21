from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import threading
from collections import Counter
from collections import namedtuple
from datetime import datetime

import numpy as np
import paths
from vocabulary import Vocabulary

FEATURES_DEF = False
VERBOSE_DEF = True
VOCAB_FILE = 'vocab.txt'
TEST_SIZE = 20000
TRAIN_SIZE = 60000
MIN_WORD_DEFAULT = 4
ImageMetadata = namedtuple("ImageMetadata",
						   ["index", "img_url", "captions"])


def _process_metadata(captions_file):
	with open(captions_file, "r") as f:
		caption_data = json.load(f)

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
	counter = Counter()
	for caption in captions:
		for word in caption:
			counter.update([word])


	print("Total words:", len(counter))

	# Filter uncommon words and sort by descending count.
	word_counts = [word for word in counter.items() if word[1] >= FLAGS.min_words]
	word_counts.sort(key=lambda x: x[1], reverse=True)
	print("Words in vocabulary:", len(word_counts))

	# Create the vocabulary dictionary.
	reverse_vocab = [item[0] for item in word_counts]
	unk_id = len(reverse_vocab)
	vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
	print(len(vocab_dict.keys()))
	reverse_vocab.append("#START#")
	reverse_vocab.append("#END#")
	vocab = Vocabulary(FLAGS.vocab_file, reverse_vocab, unk_id)

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
	j = 0
	k = 0
	for image in images:
		k += 1
		for caption in image.captions:
			encoded_caption = []
			for i, word in enumerate(caption):
				# Encode word to index
				encoded_caption.append(vocab.word_to_id(word))
			# Append encoded captions to
			encoded_caption = np.array(encoded_caption)
			input_vec = np.array([features[image.index], encoded_caption, image.img_url])
			j += 1
			if j % 100 == 0:
				print("%d/%d Appending to %s dataset image %d with url %s and caption %s" % (
				k + 1, len(images), name, image.index, image.img_url, str(caption)))
				j = 0

			dataset.append(input_vec)

	dataset = np.array(dataset)
	if name == 'training':
		size = TRAIN_SIZE
	else:
		size = TEST_SIZE
	indices = np.arange(dataset.shape[0])
	np.random.shuffle(indices)
	dataset = dataset[indices[0:size]]
	# Save processed data

	if not os.path.exists(paths.PROCESSED_FOLDER):
		os.makedirs(paths.PROCESSED_FOLDER)
	feature_file = str(feature_file).split('/')[-1]
	file_to_save = paths.PROCESSED_FOLDER + feature_file

	np.save(file_to_save, dataset)
	print("Final dataset size:\t%s" % (str(dataset.shape)), 'in ', file_to_save)


def initialize():
	'''
	Initialize the data.
	'''
	features = bool(FLAGS.features)

	print("----------------------------- Parsing targets -----------------------------")
	train_dataset = _process_metadata(paths.TRAIN_CAPTIONS_DEF)
	val_dataset = _process_metadata(paths.VALIDATION_CAPTIONS_DEF)

	if FLAGS.verbose:
		print("Training Dataset  length: %d" % (len(train_dataset)))
		print("Test     Dataset  length: %d" % (len(val_dataset)))

	print("--------------------------- Creating vocabulary ---------------------------")
	train_captions = [caption for image in train_dataset for caption in image.captions]
	vocab = _create_vocab(train_captions)

	if FLAGS.verbose:
		print("Vocabulary length:%d " % (len(vocab._vocab)))

	print("---------------------------- Parsing  features ----------------------------")
	_process_dataset('training', train_dataset, vocab, paths.TRAIN_FEATURES_DEF)
	_process_dataset('test', val_dataset, vocab, paths.VALIDATION_FEATURES_DEF)


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
	parser.add_argument('--features', type=str, default=FEATURES_DEF, help='Parse features (true/false)')
	parser.add_argument('--verbose', type=str, default=VERBOSE_DEF, help='Show further output (true/false)')
	parser.add_argument("--vocab_file", type=str, default=VOCAB_FILE, help="Output vocabulary file of word counts.")
	parser.add_argument("--min_words", type=str, default=MIN_WORD_DEFAULT,
						help="The minimum number of occurrences of each word " +
							 "in the training set for inclusion in the vocabulary.")
	FLAGS, unparsed = parser.parse_known_args()
	main(None)
