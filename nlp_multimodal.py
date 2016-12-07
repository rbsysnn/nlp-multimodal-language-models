from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from collections import namedtuple
from collections import Counter
import json
import os

import threading
import numpy as np
from paths import VALIDATION_FEATURES_DEF as VALIDATION_FEATURES_DEF
from paths import VALIDATION_CAPTIONS_DEF as VALIDATION_CAPTIONS_DEF

from paths import TRAIN_FEATURES_DEF as TRAIN_FEATURES_DEF
from paths import TRAIN_CAPTIONS_DEF as TRAIN_CAPTIONS_DEF

FEATURES_DEF = False
VERBOSE_DEF = True
VOCAB_FILE = 'vocab.txt'
MIN_WORD_DEFAULT = 4
ImageMetadata = namedtuple("ImageMetadata",
                           ["index", "img_url", "captions"])





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
        image_url = captions[0]
        captions = captions[1]
        for i, c in enumerate(captions):
            c = [word.lower() for word in c]
            captions[i] = c

        url_to_captions.setdefault(image_url, [])
        url_to_captions[image_url].append(captions)
    print(len(filenames))
    print(len(url_to_captions))
    assert (len(filenames) == len(url_to_captions))
    assert (set([x for x in filenames]) == set(url_to_captions.keys()))
    print("Loaded caption metadata for %d images from %s" %
          (len(url_to_captions), captions_file.split('/')[-1]))

    # Process the captions and combine the data into a list of ImageMetadata.
    print("Proccessing captions.")
    image_metadata = []
    num_captions = 0
    # print(url_to_captions)
    for index, url in enumerate(filenames):
        captions = [c for c in url_to_captions[url]]

        image_metadata.append(ImageMetadata(index, url, captions))
        num_captions += len(captions) * 5
    print("Finished processing %d captions for %d images in %s" %
          (num_captions, len(filenames), captions_file.split('/')[-1]))
    return image_metadata


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
        for caption_list in image.captions:
            for caption in caption_list:
                for i, word in enumerate(caption):
                    caption[i] = vocab.word_to_id(word)

                ind_caption = np.array(caption)

                input_vec = np.array([features[image.index], ind_caption])
                print("Appending to dataset image %d\t" % (image.index))
                dataset.append(input_vec)

    dataset = np.array(dataset)
    print("Final dataset size:%s" % (str(dataset.shape)))
    file_to_save = 'name_' + feature_file
    if not os.path.exists(file_to_save):
        os.makedirs(file_to_save)

    np.save(file_to_save, np.array(dataset))


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


def initialize():
    '''
    Initialize the data.
    '''
    features = bool(FLAGS.features)
    validation_captions_file = FLAGS.valid_capt
    validation_features_file = FLAGS.valid_feat
    training_features_file = FLAGS.train_feat
    training_captions_file = FLAGS.train_capt
    print("Parsing files...")
    train_dataset = _load_and_process_metadata(training_captions_file)
    val_dataset = _load_and_process_metadata(validation_captions_file)

    if FLAGS.verbose:
        print("Training Dataset  length:%d" % (len(train_dataset)))
        print("Test     Dataset  length:%d" % (len(val_dataset)))

    train_captions = [c for image in train_dataset for c in image.captions]
    vocab = _create_vocab(train_captions)

    print("Vocabulary length:%d " % (len(vocab._vocab)))
    _process_dataset('training',train_dataset, vocab, training_features_file)
    _process_dataset('test',val_dataset, vocab, validation_features_file)

def gather_captions(image_captions):
    corpus = set()
    # print("Images to parse %d"%(len(image_captions)))
    for img_index, img in enumerate(image_captions):
        url = image_captions[img_index][0]
        for i, captions in enumerate(image_captions[img_index][1]):
            for word in captions:
                corpus.add(word)
    return corpus


def print_corpus(corpus):
    print("Corpus length:%d" % (len(corpus)))
    for index, word in enumerate(corpus):
        print(str(word.encode('utf-8')))


def _write_flags_to_file(f):
    for key, value in vars(FLAGS).items():
        f.write(key + ' : ' + str(value) + '\n')


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' \t:\t ' + str(value))




def main(_):
    print("Hello")
    print_flags()
    initialize()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_shards", type=int, default=256, help="Number of shards in training TFRecord files.")
    parser.add_argument("--val_shards", type=int, default=4, help="Number of shards in validation TFRecord files.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for creating data.")
    parser.add_argument('--valid_feat', type=str, default=VALIDATION_FEATURES_DEF, help='Validation features file')
    parser.add_argument('--valid_capt', type=str, default=VALIDATION_CAPTIONS_DEF, help='Validation captions file')
    parser.add_argument('--train_feat', type=str, default=TRAIN_FEATURES_DEF, help='Training features file')
    parser.add_argument('--train_capt', type=str, default=TRAIN_CAPTIONS_DEF, help='Training captions file')
    parser.add_argument('--features', type=str, default=FEATURES_DEF, help='Parse features (true/false)')
    parser.add_argument('--verbose', type=str, default=VERBOSE_DEF, help='Show further output (true/false)')
    parser.add_argument("--vocab_file", type=str, default=VOCAB_FILE, help="Output vocabulary file of word counts.")
    parser.add_argument("--min_words", type=str, default=MIN_WORD_DEFAULT,
                        help="The minimum number of occurrences of each word " +
                             "in the training set for inclusion in the vocabulary.")
    FLAGS, unparsed = parser.parse_known_args()
    main(None)
