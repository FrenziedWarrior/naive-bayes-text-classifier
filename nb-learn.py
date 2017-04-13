# This is the script that learns a Naive Bayes model from training data (reviews)
# Reads the reviews and annotated labels from i/p files and computes model parameters training data
# i/p: dev-train-text.txt, dev-train-labels.txt

# NO_OF_REVIEWS = 1280
# TRAINING_SPLIT = 0.75
# DEV_TEST_SPLIT = 0.25

# 1 -> Read reviews line by line and perform tokenization (separate text into word tokens)
#   a: Read corresponding labels assigned to the review and store required counts
# 2 -> Maintain counts of term occurrences in documents of both classes in different data structures (dict)
# 3 -> Count total number of word tokens in documents of all classes

from math import log
import sys
import re
import operator


# dict_dt = {
#     'deceptive': 0,
#     'truthful': 0
# }
#
# dt_term_count = {
#     'deceptive': {},
#     'truthful': {}
# }
#
# dict_pn = {
#     'positive': 0,
#     'negative': 0
# }
#
# pn_term_count = {
#     'positive': {},
#     'negative': {}
# }

# four_way_labels = ['deceptive positive', 'deceptive negative', 'truthful positive', 'truthful negative']


class NaiveBayesClassifier(object):
    review_content = dict()
    entire_vocab = list()
    stop_words = list()

    def __init__(self, labels):
        self.classes = labels    # labels = ['deceptive', 'truthful']
        self.label_code = [label[0] for label in self.classes]   # label_code = ['d', 't']
        self.review_labels = dict()
        self.doc_count = {code: 0 for code in self.label_code}
        self.term_count = {code: {} for code in self.label_code}
        self.priors = {code: 0.0 for code in self.label_code}

    @classmethod
    def remove_stop_words(cls, all_terms):
        return [term for term in all_terms if term not in cls.stop_words]

    def increment_doc_count(self, curr_id):
        curr_rev_label = self.review_labels[curr_id][0]
        self.doc_count[curr_rev_label] += 1

    def update_term_freqs(self, curr_id):
        curr_rev_label = self.review_labels[curr_id][0]
        for term in self.review_content[curr_id]:
            self.term_count[curr_rev_label].setdefault(term, 0) + 1


# retrieve list of stop words
def extract_stop_words():
    fn_swl = "stop-words.txt"
    with open(fn_swl, 'r') as swl_file:
        return [sw.rstrip('\n') for sw in swl_file.readlines()]


# retrieve labels from train-labels file
def retrieve_review_labels(text):
    label_text = text.strip().split()
    rev_id, rev_annotations = label_text[0], label_text[1:]
    return rev_id, rev_annotations


# tokenization of review text
def tokenize_review_text(text):
    rev_text = text.strip(' \n\r')
    rev_text = re.sub(r'[.,;:*?&#~()!$]', r' ', rev_text)
    rev_text = rev_text.replace('\'s', ' ')
    rev_text = rev_text.split(' ')
    rev_id, rev_content = rev_text[0], rev_text[1:]
    rev_word_list = [word.strip(' ;?()~*\'\"').lower() for word in rev_content]
    rev_word_list = [word for word in rev_word_list if word != '' and not word.isdigit()]
    return rev_id, rev_word_list


# reads dataset and generates review per line
def generate_review():
    fn_review_text = sys.argv[1]
    fn_review_labels = sys.argv[2]
    with open(fn_review_text, 'r') as f_text, open(fn_review_labels, 'r') as f_labels:
        for text, labels in zip(f_text, f_labels):
            yield text, labels


# program starts here
def start_program():
    dt_filter = NaiveBayesClassifier(['deceptive', 'truthful'])
    NaiveBayesClassifier.stop_words = extract_stop_words()
    for next_review, next_labels in generate_review():
        content_rev_id, curr_rev_wordlist = tokenize_review_text(next_review)
        cleaned_rev_wordlist = NaiveBayesClassifier.remove_stop_words(curr_rev_wordlist)
        dt_filter.review_content[content_rev_id] = cleaned_rev_wordlist
        labels_rev_id, curr_rev_labels = retrieve_review_labels(next_labels)    # [0: 'd'/'t', 1: 'p','n']
        dt_filter.review_labels[labels_rev_id] = curr_rev_labels[0]
        # pn_filter.review_labels[labels_rev_id] = curr_rev_labels[1]

        # assumed that reviews and labels are sorted acc to hash-id
        assert content_rev_id == labels_rev_id

        dt_filter.increment_doc_count(content_rev_id)
        dt_filter.update_term_freqs(content_rev_id)


start_program()
# dec_prior = log(dict_dt['deceptive'] / (dict_dt['deceptive'] + dict_dt['truthful']))
# tru_prior = log(dict_dt['truthful'] / (dict_dt['deceptive'] + dict_dt['truthful']))
# pos_prior = log(dict_pn['positive'] / (dict_pn['positive'] + dict_pn['negative']))
# neg_prior = log(dict_pn['negative'] / (dict_pn['positive'] + dict_pn['negative']))
#
# dec_word_list = dt_term_count['deceptive']
# tru_word_list = dt_term_count['truthful']
# pos_word_list = pn_term_count['positive']
# neg_word_list = pn_term_count['negative']

# sorted_freq1 = dict(sorted(dec_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
# sorted_freq2 = dict(sorted(tru_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
# sorted_freq3 = dict(sorted(pos_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
# sorted_freq4 = dict(sorted(neg_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
# print(sorted_freq1)
# print(sorted_freq2)
# print(sorted_freq3)
# print(sorted_freq4)

# list_vocab = set(
#     list(dec_word_list.keys()) + list(tru_word_list.keys())
# )
#
# for word in list_vocab:
#     if word in dec_word_list:
#         dec_token = True
#         rc_dec = dec_word_list[word]
#     else:
#         dec_token = False
#         rc_dec = 0
#
#     if word in tru_word_list:
#         tru_token = True
#         rc_tru = tru_word_list[word]
#     else:
#         tru_token = False
#         rc_tru = 0
#
#     # removing rare tokens
#     if rc_tru + rc_dec < 3:
#         if dec_token:
#             dec_word_list.pop(word)
#         elif tru_token:
#             tru_word_list.pop(word)
#
# # counting length of documents of each class
# len_dec_docs = 0
# len_tru_docs = 0
# len_pos_docs = 0
# len_neg_docs = 0
#
# for key, val in dec_word_list.items():
#     len_dec_docs += val
#
# for key, val in tru_word_list.items():
#     len_tru_docs += val
#
# for key, val in pos_word_list.items():
#     len_pos_docs += val
#
# for key, val in neg_word_list.items():
#     len_neg_docs += val
#
# sorted_vocab = sorted(set(
#     list(dec_word_list.keys()) + list(tru_word_list.keys())
# ))
#
# for word in sorted_vocab:
#     # Word Token Type
#     target_model.write(word + ' ')
#     #    print("%20s" % word, end='\t\t')
#
#     # Word Token raw count - (deceptive)
#     if word in dec_word_list:
#         target_model.write(str(dec_word_list[word]) + ' ')
#     # print(str(dec_word_list[word]), end=' ')
#     else:
#         target_model.write(str(0) + ' ')
#     # print(str(0), end=' ')
#
#     # Word Token raw count - (truthful)
#     if word in tru_word_list:
#         target_model.write(str(tru_word_list[word]) + ' ')
#     # print(str(tru_word_list[word]), end=' ')
#     else:
#         target_model.write(str(0) + ' ')
#     # print(str(0), end=' ')
#
#     # Word Token raw count - (positive)
#     if word in pos_word_list:
#         target_model.write(str(pos_word_list[word]) + ' ')
#     # print(str(pos_word_list[word]), end=' ')
#     else:
#         target_model.write(str(0) + ' ')
#     # print(str(0), end=' ')
#
#     # Word Token raw count - (negative)
#     if word in neg_word_list:
#         target_model.write(str(neg_word_list[word]) + ' ')
#     # print(str(neg_word_list[word]))
#     else:
#         target_model.write(str(0) + ' ')
#     # print(str(0))
#
#     # P(Tk/C) probability - (deceptive)
#     if word in dec_word_list:
#         target_model.write(str(log(dec_word_list[word] / len_dec_docs)) + ' ')
#     else:
#         target_model.write(str(0) + ' ')
#
#     # P(Tk/C) probability - (truthful)
#     if word in tru_word_list:
#         target_model.write(str(log(tru_word_list[word] / len_tru_docs)) + ' ')
#     else:
#         target_model.write(str(0) + ' ')
#
#     # P(Tk/C) probability - (positive)
#     if word in pos_word_list:
#         target_model.write(str(log(pos_word_list[word] / len_pos_docs)) + ' ')
#     else:
#         target_model.write(str(0) + ' ')
#
#     # P(Tk/C) probability - (negative)
#     if word in neg_word_list:
#         target_model.write(str(log(neg_word_list[word] / len_neg_docs)) + ' ')
#     else:
#         target_model.write(str(0) + ' ')
#
#     # log P(Tk/C) probability after smoothing - (deceptive)
#     if word in dec_word_list:
#         target_model.write(str(log((dec_word_list[word] + 1) / (len_dec_docs + len(sorted_vocab)))) + ' ')
#     else:
#         target_model.write(str(log(1 / (len_dec_docs + len(sorted_vocab)))) + ' ')
#
#     # log P(Tk/C) probability after smoothing - (truthful)
#     if word in tru_word_list:
#         target_model.write(str(log((tru_word_list[word] + 1) / (len_tru_docs + len(sorted_vocab)))) + ' ')
#     else:
#         target_model.write(str(log(1 / (len_dec_docs + len(sorted_vocab)))) + ' ')
#
#     # log P(Tk/C) probability after smoothing - (positive)
#     if word in pos_word_list:
#         target_model.write(str(log((pos_word_list[word] + 1) / (len_pos_docs + len(sorted_vocab)))) + ' ')
#     else:
#         target_model.write(str(log(1 / (len_pos_docs + len(sorted_vocab)))) + ' ')
#
#     # log P(Tk/C) probability after smoothing - (negative)
#     if word in neg_word_list:
#         target_model.write(str(log((neg_word_list[word] + 1) / (len_neg_docs + len(sorted_vocab)))) + '\n')
#     else:
#         target_model.write(str(log(1 / (len_neg_docs + len(sorted_vocab)))) + '\n')
