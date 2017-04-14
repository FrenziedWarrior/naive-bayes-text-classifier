# This is the script that learns a Naive Bayes model from training data (reviews)
# Reads the reviews and annotated labels from i/p files and computes model parameters training data
# i/p: dev-train-text.txt, dev-train-labels.txt

# NO_OF_REVIEWS = 1280

# 1 -> Read reviews line by line and perform tokenization (separate text into word tokens)
#   a: Read corresponding labels assigned to the review and store required counts
# 2 -> Maintain counts of term occurrences in documents of both classes in different data structures (dict)
# 3 -> Count total number of word tokens in documents of all classes

import math
import sys
import re


class NaiveBayesClassifier(object):
    review_content = dict()
    feature_vocab = list()
    stop_words = list()

    def __init__(self, labels):
        self.classes = labels    # labels = ['deceptive', 'truthful']
        self.label_code = [label[0] for label in self.classes]   # label_code = ['d', 't']
        self.review_labels = dict()
        self.doc_count = {code: 0 for code in self.label_code}
        self.doc_length = {code: 0 for code in self.label_code}
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

    def calculate_priors(self):
        sum_docs = sum(v for v in self.doc_count.values())
        for k in self.priors.keys():
            self.priors[k] = math.log(self.doc_count[k]/sum_docs)

    def construct_vocabulary(self):
        codes = self.label_code
        for word in sorted(set(k for cl in codes for k in self.term_count[cl].keys())):
            if sum(freq for freq in [class_dict.get(word, 0)
                         for class_dict in self.term_count.values()]
                   ) > 3:
                self.feature_vocab.append(word)

    def calculate_document_lengths(self):
        for k in self.doc_length.keys():
            self.doc_length[k] = sum(val for key, val in self.term_count[k].items() if key in self.feature_vocab)


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

    dt_filter.calculate_priors()
    dt_filter.construct_vocabulary()
    dt_filter.calculate_document_lengths()

start_program()

# import operator: use in sorted() to sort dictionary by values
# to find n most frequently occuring words in each class
# sorted_freq1 = dict(sorted(class_word_list.items(), key=operator.itemgetter(1), reverse=True)[:15])
#
# for word in sorted_vocab:
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
