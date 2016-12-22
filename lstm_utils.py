
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from PIL import Image
import urllib
class DataSet(object):
	"""
	Utility class to handle dataset structure.
	"""

	def __init__(self, features, captions,urls):
		"""
		Builds dataset with features and captions.
		Args:
			features: features data.
			captions: captions data
		"""
		assert features.shape[0] == captions.shape[0] == urls.shape[0], (
					"features.shape: {0}, captions.shape: {1},urls.shape{2}".format(str(features.shape), str(captions.shape),str(urls.shape)))

		self._num_examples = features.shape[0]
		self._features = features
		self._urls = urls
		self._captions = captions
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def features(self):
		return self._features

	@property
	def captions(self):
		return self._captions

	@property
	def urls(self):
			return self._urls
	
	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""
		Return the next `batch_size` examples from this data set.

		Args:
			batch_size: Batch size.

		Returns:
			features: features data.
			captions: captions data.

		"""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1

			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._features = self._features[perm]
			self._captions = self._captions[perm]
			self._urls     = self._urls[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch
		return self._features[start:end], self._captions[start:end],self._urls[start:end]

def get_num_classes(filename):
	with open(filename) as f:
				for i, l in enumerate(f):
						pass
	return i + 1

def dense_to_one_hot(captions_dense,num_classes):
	"""
	Convert class captions from scalars to one-hot vectors.
	Args:
		captions_dense: Dense captions.

	Outputs:
		captions_one_hot: One-hot encoding for captions.
	"""
	num_captions = captions_dense.shape[0]

	encoded = []

	for i,caption in enumerate(captions_dense):
				caption_length = len(caption)
				encoded_caption = np.zeros((caption_length,num_classes),dtype='int32')
				index_offset = np.arange(caption_length) * num_classes
				encoded_caption.flat[index_offset + caption.ravel()] = 1
				encoded.append(encoded_caption)
				if i % 10 == 0:
					print(i)
	return np.array(encoded)
def read_data_sets(data_dir, validation_size,max_length):
	"""
	Returns the dataset readed from data_dir.
	Uses or not uses one-hot encoding for the captions.
	Subsamples validation set with specified size if necessary.

	Args:
		data_dir: Data directory.
		validation_size: Size of validation set

	Returns:
		Train, Validation, Test Datasets
	"""
	dataset = np.load(data_dir + "merged_train.npy")

	features = []
	captions = []
	urls = []
	j = 0
	while j < len(dataset):
		if len(dataset[j][1]) > max_length:
			j += 1
			continue
		feature = dataset[j][0]
		caption = dataset[j][1]
		url 	= dataset[j][2]
		features.append(feature)
		captions.append(caption)
		urls.append(url)
		j +=1 

	train_features = np.array(features)
	train_captions = np.array(captions)
	train_urls 	   = np.array(urls)
	dataset = np.load(data_dir + "merged_val.npy")
	features = []
	captions = []
	urls = []
	j = 0
	while j < len(dataset):
		if len(dataset[j][1]) > max_length: 
			j += 1
			continue
		feature = dataset[j][0]
		caption = dataset[j][1]
		url 	= dataset[j][2]
		features.append(feature)
		captions.append(caption)
		urls.append(url)
		j +=1

	test_features = np.array(features)
	test_captions = np.array(captions)
	test_urls     = np.array(urls)
	if not os.path.exists(data_dir+'/'+str(max_length)):
		os.mkdir(data_dir+'/'+str(max_length))


	np.save(data_dir+'/'+str(max_length)+'/train_features.npy',train_features)
	np.save(data_dir+'/'+str(max_length)+'/train_captions.npy',train_captions)
	np.save(data_dir+'/'+str(max_length)+'/test_features.npy',test_features)
	np.save(data_dir+'/'+str(max_length)+'/test_captions.npy',test_captions)
	np.save(data_dir+'/'+str(max_length)+'/train_urls.npy',train_urls)
	np.save(data_dir+'/'+str(max_length)+'/test_urls.npy',test_urls)
	print('========================== LOADED features =======================================')
	print(train_features.shape[0],"\t",test_features.shape[0],"Test and train size")
	# Apply one-hot encoding if specified
	# if one_hot:
	#   print('apply one-hot encoding')
	#   num_classes =  get_num_classes(filename='vocab.txt')
	#   train_captions = dense_to_one_hot(train_captions, num_classes)
	#   test_captions = dense_to_one_hot(test_captions, num_classes)




	# Subsample the validation set from the train set
	if not 0 <= validation_size <= len(train_features):
		raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
				len(train_features), validation_size))

	validation_urls     = train_urls[:validation_size]
	validation_features = train_features[:validation_size]
	validation_captions = train_captions[:validation_size]
	train_urls     = train_urls[validation_size:]
	train_features = train_features[validation_size:]
	train_captions = train_captions[validation_size:]

	# Create datasets
	train = DataSet(train_features, train_captions,train_urls)
	validation = DataSet(validation_features, validation_captions,validation_urls)
	test = DataSet(test_features, test_captions,test_urls)
	return train,test,validation


def get_merged(data_dir = 'datasets/processed/', validation_size = 0,max_length=29):
	"""
	Prepares CIFAR10 dataset.

	Args:
		data_dir: Data directory.
		validation_size: Size of validation set

	Returns:
		Train, Validation, Test Datasets
	"""
	return read_data_sets(data_dir, validation_size,max_length)


def get_prepared_merged(data_dir = './datasets/processed', validation_size = 0,max_length=29,filename='vocab.txt'):
	

	test_features =  np.load(data_dir+'/'+str(max_length)+'/test_features.npy')
	test_captions =  np.load(data_dir+'/'+str(max_length)+'/test_captions.npy')
	test_urls     =  np.load(data_dir+'/'+str(max_length)+'/test_urls.npy')
	train_features = np.load(data_dir+'/'+str(max_length)+'/train_features.npy')
	train_captions = np.load(data_dir+'/'+str(max_length)+'/train_captions.npy')
	train_urls     = np.load(data_dir+'/'+str(max_length)+'/train_urls.npy')

	print('========================== LOADED ready features of length %d ======================================='%(max_length))
	print(train_features.shape[0],"\t",test_features.shape[0],"Test and train size")
	# Apply one-hot encoding if specified
	# if one_hot:
	#   print('apply one-hot encoding')
	#   num_classes =  get_num_classes(filename)
	#   train_captions = dense_to_one_hot(train_captions, num_classes)
	#   test_captions = dense_to_one_hot(test_captions, num_classes)


	# Subsample the validation set from the train set
	if not 0 <= validation_size <= len(train_features):
		raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
				len(train_features), validation_size))

	validation_urls     = train_urls[:validation_size]
	validation_features = train_features[:validation_size]
	validation_captions = train_captions[:validation_size]
	train_urls     = train_urls[validation_size:]
	train_features = train_features[validation_size:]
	train_captions = train_captions[validation_size:]

	# Create datasets
	train = DataSet(train_features, train_captions,train_urls)
	validation = DataSet(validation_features, validation_captions,validation_urls)
	test = DataSet(test_features, test_captions,test_urls)
	return train,test,validation



	# def get_images(urls,captions):

	# for i,url in enumerate(urls):
	#   img_id = url.split("/")[4]
	#   img_id = img_id.split("_")[2]
	#   img_id = img_id.split(".")[0]
	#   fetch_url = 'http://mscoco.org/images/'+id
	#   img_array = Image.open(urllib(fetch_url,img_id))