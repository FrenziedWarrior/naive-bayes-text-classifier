# This is the script that learns a Naive Bayes model from training data(reviews) and writes output to nbmodel.txt
# i/p: train-text.txt, train-labels.txt
# o/p: nbmodel.txt
# Reads the reviews and annotated labels from i/p files and computes model parameters training data

#NO_OF_REVIEWS = 1280
#TRAINING_SPLIT = 0.75
#DEV_TEST_SPLIT = 0.25

#1 -> Read reviews line by line and perform tokenization (separate text into word tokens)
#1a -> Read corresponding labels assigned to the review and store required counts
#2 -> Maintain counts of term occurrences in documents of both classes in different data structures (dict)
#3 -> Count total number of word tokens in documents of all classes
#4 -> 

from math import log, log2
import sys
import operator

REVIEWS_FILE_PATH = sys.argv[1]
LABELS_FILE_PATH = sys.argv[2]
MODEL_FILE_PATH = 'nbmodel.txt'

dict_dt = {'deceptive': 0, 'truthful': 0}
dt_term_count = { 'deceptive': {}, 'truthful': {} }
dict_pn = {'positive': 0, 'negative': 0}
pn_term_count = { 'positive': {}, 'negative': {} }

#class_doubles = ['deceptive positive', 'deceptive negative', 'truthful positive', 'truthful negative']

dict_reviews = {}
dict_labels = {}
list_vocab = []

stop_word_list = ['and', 'a', 'an', 'the', 'but', 'yet', 'or', 'so', 'also', 'of', 'i', 'to', 'it', 'for', 'was', 'in', 'we',
                 'is', 'at', 'my', 'that', 'on', 'our', 'this', 'they', 'he', 'she', 'had', 'were', 'with', 'be', 'you', 'are',
                 'from', 'there', 'as', 'have', 'when', 'would', 'my', 'me', 'did', 'us','chicago']
                 
def filter_stop_words(l):
    return [item for item in l if item not in stop_word_list]
    

with open(REVIEWS_FILE_PATH, 'r') as f_text, open(LABELS_FILE_PATH, 'r') as f_labels:
    for review_text, review_labels in zip(f_text, f_labels):
        # Tokenization and Label Identification
        review_text = review_text.strip(' \n\r')
        review_text = review_text.replace('.', ' ')
        review_text = review_text.replace('!', ' ')
        review_text = review_text.replace('$', ' ')
        review_text = review_text.replace('\'s', ' ')
        review_text = review_text.replace(',', ' ')
        review_text = review_text.replace('--', ' ')
#        review_text = review_text.replace('-', ' ')
        review_text = review_text.replace('&', ' ')
#        review_text = review_text.replace(':', ' ')

        review_id_text = review_text.split(' ')[0]
        review_text = review_text.split(' ')[1:]
        wordlist = [word.strip(' ;?()~*\'\"').lower() for word in review_text]
        wordlist = [word for word in wordlist if word!='' and not word.isdigit()]
        
        wordlist = filter_stop_words(wordlist)
        dict_reviews[review_id_text] = wordlist

        # Retrieving labels
        review_labels = review_labels.strip(' \n\r').split()
        review_id_labels = review_labels[0]
        review_annotations = review_labels[1:]
        dict_labels[review_id_labels] = review_annotations
        dict_dt[dict_labels[review_id_labels][0]] += 1  # increasing count for the corresponding class
        dict_pn[dict_labels[review_id_labels][1]] += 1  # increasing count for the corresponding class    
        dt_vocab = dt_term_count[review_annotations[0]]    # for deceptive/truthful
        pn_vocab = pn_term_count[review_annotations[1]]    # for positive/negative
        for term in dict_reviews[review_id_text]:         # frequency distribution table out of the words in the text
            if term not in dt_vocab:
                dt_vocab[term] = 1
            else:
                dt_vocab[term] += 1

            if term not in pn_vocab:
                pn_vocab[term] = 1
            else:
                pn_vocab[term] += 1



dec_prior = log(dict_dt['deceptive']/(dict_dt['deceptive'] + dict_dt['truthful']))
tru_prior = log(dict_dt['truthful']/(dict_dt['deceptive'] + dict_dt['truthful']))
pos_prior = log(dict_pn['positive']/(dict_pn['positive'] + dict_pn['negative']))
neg_prior = log(dict_pn['negative']/(dict_pn['positive'] + dict_pn['negative']))

dec_word_list = dt_term_count['deceptive']
tru_word_list = dt_term_count['truthful']
pos_word_list = pn_term_count['positive']
neg_word_list = pn_term_count['negative']

target_model = open(MODEL_FILE_PATH, 'w')
target_model.write('P(\'deceptive\') ' + str(dict_dt['deceptive']) + ' ' + str(dec_prior) + '\n')
target_model.write('P(\'truthful\') ' + str(dict_dt['truthful']) + ' ' + str(tru_prior) + '\n')
target_model.write('P(\'positive\') ' + str(dict_pn['positive']) + ' ' + str(pos_prior) + '\n')
target_model.write('P(\'negative\') ' + str(dict_pn['negative']) + ' ' + str(neg_prior) + '\n')


#sorted_freq1 = dict(sorted(dec_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
#sorted_freq2 = dict(sorted(tru_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
#sorted_freq3 = dict(sorted(pos_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
#sorted_freq4 = dict(sorted(neg_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
#print(sorted_freq1) 
#print(sorted_freq2) 
#print(sorted_freq3) 
#print(sorted_freq4)

list_vocab = set(list(dec_word_list.keys()) + list(tru_word_list.keys()))

for word in list_vocab:
    if word in dec_word_list:
        dec_token = True
        rc_dec = dec_word_list[word]
    else:
        dec_token = False
        rc_dec = 0
    

    if word in tru_word_list:
        tru_token = True
        rc_tru = tru_word_list[word]
    else:
        tru_token = False
        rc_tru = 0
    
    # removing rare tokens
    if rc_tru + rc_dec < 3:
        if dec_token:
            dec_word_list.pop(word)
        elif tru_token:
            tru_word_list.pop(word)


# counting length of documents of each class
len_dec_docs = 0
len_tru_docs = 0
len_pos_docs = 0
len_neg_docs = 0

for key, val in dec_word_list.items():
    len_dec_docs += val

for key, val in tru_word_list.items():
    len_tru_docs += val

for key, val in pos_word_list.items():
    len_pos_docs += val

for key, val in neg_word_list.items():
    len_neg_docs += val


sorted_vocab = sorted(set(list(dec_word_list.keys()) + list(tru_word_list.keys())))

for word in sorted_vocab:
    # Word Token Type 
    target_model.write(word + ' ')
#    print("%20s" % word, end='\t\t')

    # Word Token raw count - (deceptive)
    if word in dec_word_list:
        target_model.write(str(dec_word_list[word]) + ' ')
#        print(str(dec_word_list[word]), end=' ')
    else:
        target_model.write(str(0) + ' ')
#        print(str(0), end=' ')

    # Word Token raw count - (truthful)
    if word in tru_word_list:
        target_model.write(str(tru_word_list[word]) + ' ')
#        print(str(tru_word_list[word]), end=' ')
    else:
        target_model.write(str(0) + ' ')
#        print(str(0), end=' ')

    # Word Token raw count - (positive)
    if word in pos_word_list: 
        target_model.write(str(pos_word_list[word]) + ' ')
#        print(str(pos_word_list[word]), end=' ')
    else:
        target_model.write(str(0) + ' ')
#        print(str(0), end=' ')


    # Word Token raw count - (negative)
    if word in neg_word_list: 
        target_model.write(str(neg_word_list[word]) + ' ')
#        print(str(neg_word_list[word]))
    else:
        target_model.write(str(0) + ' ')
#        print(str(0))
    
    # P(Tk/C) probability - (deceptive)
    if word in dec_word_list:
        target_model.write(str(log(dec_word_list[word]/len_dec_docs)) + ' ')
    else:
        target_model.write(str(0) + ' ')

        
    # P(Tk/C) probability - (truthful)
    if word in tru_word_list: 
        target_model.write(str(log(tru_word_list[word]/len_tru_docs)) + ' ')
    else:
        target_model.write(str(0) + ' ')
    
    # P(Tk/C) probability - (positive)
    if word in pos_word_list: 
        target_model.write(str(log(pos_word_list[word]/len_pos_docs)) + ' ')
    else:
        target_model.write(str(0) + ' ')
    
    # P(Tk/C) probability - (negative)
    if word in neg_word_list: 
        target_model.write(str(log(neg_word_list[word]/len_neg_docs)) + ' ')
    else:
        target_model.write(str(0) + ' ')
    

    # log P(Tk/C) probability after smoothing - (deceptive)
    if word in dec_word_list:
        target_model.write(str(log( (dec_word_list[word]+1) / (len_dec_docs+len(sorted_vocab)) )) + ' ')
    else:
        target_model.write(str(log( 1/(len_dec_docs+len(sorted_vocab)) )) + ' ')

        
    # log P(Tk/C) probability after smoothing - (truthful)
    if word in tru_word_list: 
        target_model.write(str(log( (tru_word_list[word]+1) / (len_tru_docs+len(sorted_vocab)) )) + ' ')
    else:
        target_model.write(str(log( 1/(len_dec_docs+len(sorted_vocab)) )) + ' ')
    
    # log P(Tk/C) probability after smoothing - (positive)
    if word in pos_word_list:
        target_model.write(str(log( (pos_word_list[word]+1) / (len_pos_docs+len(sorted_vocab)) )) + ' ')
    else:
        target_model.write(str(log( 1/(len_pos_docs+len(sorted_vocab)) )) + ' ')

        
    # log P(Tk/C) probability after smoothing - (negative)
    if word in neg_word_list: 
        target_model.write(str(log( (neg_word_list[word]+1) / (len_neg_docs+len(sorted_vocab)) )) + '\n')
    else:
        target_model.write(str(log( 1/(len_neg_docs+len(sorted_vocab)) )) + '\n')
    
    
    
    
    
    
    
