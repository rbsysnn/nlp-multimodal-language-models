from __future__ import division

import nltk

def bleu_score(reference,hypothesis):
#hypothesis=output_caption
#references=img_captions        
#hypothesis = ['It', 'is', 'a', 'cat', 'in','a', 'room']
#references = [['There', 'is', 'a', 'cat', 'in', 'the', 'room'],['There', 'is', 'a', 'cat', 'in', 'the', 'room']]
	
	#reference always as a list of list [reference]
	#bleu_score
	bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
	#bleu_score as 4-gram
	bleu_score_4gram=nltk.translate.bleu_score.modified_precision([reference], hypothesis,4)
	
	return bleu_score,bleu_score_4gram

		
