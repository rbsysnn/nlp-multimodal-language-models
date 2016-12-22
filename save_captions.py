
# coding: utf-8

# In[3]:

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


# In[4]:

def _save_captions_to_file(predictions,targets):
    if not os.path.exists(paths.PROCESSED_FOLDER):
        os.makedirs(paths.PROCESSED_FOLDER)

    file_to_save = paths.PROCESSED_FOLDER + "results.npy"
    
    pred_indices = np.argmax(predictions,axis=2)
    for i in range(pred_indices):
        captions[i] = [ _map_to_sentence(predictions[i]),  _map_to_sentence(targets[i])]     
    np.save(file_to_save, captions)

