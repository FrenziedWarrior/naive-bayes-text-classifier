# This is the script that classifies test data(reviews) and assigns labels using Naive Bayes assumption
# i/p: nbmodel.txt
# o/p: nboutput.txt
# Reads the model parameters learnt from the training data, and calculates C_MAP for different classes

import sys 

MODEL_FILE = 'nbmodel.txt'
OUTPUT_FILE_PATH = 'nboutput.txt'
TEST_FILE = sys.argv[1]

dict_all_words = {}

f_model = open(MODEL_FILE, 'r')
f_test = open(TEST_FILE, 'r')
f_op = open(OUTPUT_FILE_PATH, 'w')

dec_prior = f_model.readline().strip('\n\r').split(' ')[-1]
tru_prior = f_model.readline().strip('\n\r').split(' ')[-1]
pos_prior = f_model.readline().strip('\n\r').split(' ')[-1]
neg_prior = f_model.readline().strip('\n\r').split(' ')[-1]

for line in f_model:
    l_param = line.strip('\n\r').split(' ')
    dict_all_words[l_param[0]] = [float(val) for val in l_param[-4:]]
    

for line in f_test:
    review_text = line.strip(' \n\r')
    review_text = review_text.replace('.', ' ')
    review_text = review_text.replace('!', ' ')
    review_text = review_text.replace('$', ' ')
    review_text = review_text.replace('\'s', ' ')
    review_text = review_text.replace(',', ' ')
    review_text = review_text.replace('--', ' ')
    review_id_text = review_text.split(' ')[0]
    review_text = review_text.split(' ')[1:]

    review_word_list = [word.strip(' ;?()*\'\"').lower() for word in review_text]
    review_word_list = [word for word in review_word_list if word!='' and not word.isdigit()]
    
    dec_score = float(dec_prior)
    tru_score = float(tru_prior)
    pos_score = float(pos_prior)
    neg_score = float(neg_prior)
    for word in review_word_list:
        if word in dict_all_words:
            dec_score += dict_all_words[word][0]
            tru_score += dict_all_words[word][1]
            pos_score += dict_all_words[word][2]
            neg_score += dict_all_words[word][3]
    
    if dec_score > tru_score:
        result_dt = 'deceptive'
    else:
        result_dt = 'truthful'
        
    if pos_score > neg_score:
        result_pn = 'positive'
    else:
        result_pn = 'negative'
        
    f_op.write(review_id_text + ' ' + result_dt + ' ' + result_pn + '\n')
