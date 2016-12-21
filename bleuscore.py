import nltk


def bleu_score(reference,hypothesis):
#hypothesis=output_caption
#references=img_captions        
#hypothesis = ['It', 'is', 'a', 'cat', 'in','a', 'room']
#references = [['There', 'is', 'a', 'cat', 'in', 'the', 'room'],['There', 'is', 'a', 'cat', 'in', 'the', 'room']]
    

    #there may be several references
    #reference always as a list of list [reference]
    #bleu_score
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    print bleu_score
    #bleu_score as 4-gram
    bleu_score_4gram=nltk.translate.bleu_score.modified_precision([reference], hypothesis,4)
    print bleu_score_4gram
    
    return bleu_score,bleu_score_4gram

    
# To see how it works
#b1,b2=bleu_score(['It', 'is', 'a', 'cat', 'in','a', 'room'],['It', 'is', 'a', 'cat', 'in','the', 'room'])