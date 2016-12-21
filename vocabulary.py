from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Vocabulary(object):
	"""Simple vocabulary wrapper."""




	def __init__(self,vocab_file='vocab.txt',vocab=None, unk_id=None,flag='write'):
		"""Initializes the vocabulary.
		Args:
			vocab: A dictionary of word to word_id.
			unk_id: Id of the special 'unknown' word.
		"""

			# Write out the words file.
		if flag == 'write':
			with open(vocab_file, "w") as f:
				# f.write("\n".join(["%s %d" % (word, count) for word, count in word_counts]))
				f.write("\n".join(['%s'%(word) for word in vocab]))
			print("Wrote vocabulary file:", vocab_file)
			self._vocab = vocab
			self._unk_id = unk_id
		elif flag =='load':
			vocab = []
			with open(vocab_file,'r') as f:
				for line in f.readlines():
					vocab.append(line.strip())
			print("Loaded vocabulary file:", vocab_file)
			self._vocab  = vocab
			self._unk_id = len(vocab)

		

	def word_to_id(self, word):
		"""Returns the integer id of a word string."""
		if word in self._vocab:
			# print(word,'id:',self._vocab.index(word))
			return self._vocab.index(word)
		else:
			# print('unknown')
			return self._unk_id

	def id_to_word(self, word_id):
		"""Returns the word string of an integer word id."""
		if word_id >= len(self._vocab):
			return self._vocab[self._unk_id]
		else:
			return self._vocab[word_id]

