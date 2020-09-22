"""
Language Classifier.

Not meant for real world usage, just for understanding purposes.
The Additional Dataset is a novice data set from various sources,
just for understanding purposes.
That data set has terrible results at only 58% 
"""
from helper_code import *
import pickle as pkl
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from collections import defaultdict
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import collections
from termcolor import cprint
import re
from os import get_terminal_size
plt.style.use('ggplot')

# ==========================TESTING================================== #
# # Importing a pre-trained model.
# model = joblib.load('Data/Models/final_model.joblib')
# # Loading a vectorizer that assign vectors to words.
# vectorizer = joblib.load('Data/Vectorizers/final_model.joblib')
# # TEXT is from Slovak language
# text = 'okrem iného ako durič na brlohárenie'
# # Formats the text
# text_processed = preprocess_function(text)
# # Seperates the text into subwords
# text_split = [split_into_subwords_function(text_processed)]
# # Assigns vector values to subwords.
# text_vectorized = vectorizer.transform(text_split)
# # Predicts the language
# print(f'\nLanguage of \'{text}\'=', model.predict(text_vectorized))
# ============================TESTED================================== #


def open_file(filename):
    """File Reader."""
    with open(filename, 'r', encoding='UTF-8') as file:
        data = file.readlines()
    return data


# Initializing Dictionary To Store Raw Language Data
data_raw = dict()


# Loads Data Into Dictionaries.
def load(files, languages, name=data_raw):
    """Load data into memory (in dictionaries)."""
    for file, language in zip(files, languages):
        name[language] = open_file(file)


# Loading Data Into Memory
files = ['Data/Sentences/train_sentences.sk',
         'Data/Sentences/train_sentences.cs',
         'Data/Sentences/train_sentences.en']
languages = ['Slovak', 'Czech', 'English']
# languages = ['Slovak', 'Czech', 'English', 'German']
# files = ['Additional Data/Language Sentences/Slovak.txt',
#          'Additional Data/Language Sentences/Czech.txt',
#          'Additional Data/Language Sentences/English.txt',
#          'Additional Data/Language Sentences/German.txt']
load(files, languages)


def show_statistics(data):
    """Display Data Statistics."""
    for language, sentences in data.items():

        number_of_sentences = len(sentences)
        word_list = ' '.join(sentences).split()
        number_of_words = len(word_list)
        number_of_unique_words = len(set(word_list))
        sample_extract = " ".join(word_list[:7])

        cprint(f' {language} '.center(get_terminal_size().columns - 5, "-"),
               'red')
        cprint(f'Number of sentences\t:\t {number_of_sentences}', 'green', attrs=['bold'])
        cprint(
            f'Number of words\t:\t {number_of_words}', 'green', attrs=['bold'])
        cprint(f'Number of unique words\t:\t {number_of_unique_words}', 'green', attrs=[
              'bold'])
        cprint(
            f'Sample extract\t\t:\t {sample_extract}...\n', 'green', attrs=['bold'])


def preprocess(text):
    '''
    Removes punctuation and digits from a string, and converts all characters to lowercase. 
    Also clears all \n and hyphens (splits hyphenated words into two words).
    
    '''
    preprocessed_text = text.lower().replace('-', ' ').replace('–', ' ')

    translation_table = str.maketrans(
        '\n', ' ', string.punctuation+string.digits)

    preprocessed_text = preprocessed_text.translate(translation_table)

    return preprocessed_text


data_preprocessed = {k: [preprocess(sentence) for sentence in v]
                  for k, v in data_raw.items()}


# ====================================Shows Data Statistics====================================== #
# print('RAW\n\n')
# show_statistics(data_raw)
# print('\n\nPREPROCESSED\n\n')
# show_statistics(data_preprocessed)
# do_law_of_zipf(data_raw)
# do_law_of_zipf(data_preprocessed)
# ============================================================================================== #

# Sentences, Labels
sentences_train, y_train = [], []

for k, v in data_preprocessed.items():
    for sentence in v:
        # Sentence
        sentences_train.append(sentence)
        # Label
        y_train.append(k)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(sentences_train)

# ==========================TESTING================================== #
# naive_classifier = MultinomialNB()
# naive_classifier.fit(x_train, y_train)
# MultinomialNB(alpha=1.0, class_prior=None, fit_prior=None)
# ============================TESTED================================== #

data_val = dict()

files = ['Data/Sentences/val_sentences.sk',
         'Data/Sentences/val_sentences.cs',
         'Data/Sentences/val_sentences.en']
languages = ['Slovak', 'Czech', 'English']
# languages = ['Slovak', 'Czech', 'English', 'German']
# files = ['Additional Data/Validation/Slovak.txt',
#          'Additional Data/Validation/Czech.txt',
#          'Additional Data/Validation/English.txt',
#          'Additional Data/Validation/German.txt']
load(files, languages, data_val)


data_val_preprocessed = {
    k: [preprocess(sentence) for sentence in v] for k, v in data_val.items()}

sentences_val, y_val = [], []

for k, v in data_val_preprocessed.items():
    for sentence in v:
        sentences_val.append(sentence)
        y_val.append(k)

X_val = vectorizer.transform(sentences_val)
# predictions = naive_classifier.predict(X_val)

# ======================Performance Review =========================== #
# plot_confusion_matrix(y_val, predictions, ['Slovak', 'Czech', 'English', 'German'])
# print(f1_score(y_val, predictions, average='weighted'))
# ==================================================================== #

naive_classifier = MultinomialNB(alpha=1, fit_prior=False)
naive_classifier.fit(x_train, y_train)

predictions = naive_classifier.predict(X_val)
# ======================Performance Review =========================== #
# plot_confusion_matrix(y_val, predictions, ['Slovak', 'Czech', 'English', 'German'])
# print(f1_score(y_val, predictions, average='weighted'))
# ==================================================================== #


# =================taken from/closely based on https://arxiv.org/abs/1508.07909=========== #
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_vocab(data):

    words = []
    for sentence in data:
        words.extend(sentence.split())

    vocab = defaultdict(int)
    for word in words:
        vocab[' '.join(word)] += 1

    return vocab


vocab = get_vocab(sentences_train)

# also taken from original paper
for i in range(100):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

merges = defaultdict(int)
for k, v in vocab.items():
    for subword in k.split():
        if len(subword) >= 2:
            merges[subword] += v

merge_ordered = sorted(merges, key=merges.get, reverse=True)

pkl.dump(merge_ordered, open('Additional Data/merge_ordered.pkl', 'wb'))
# ============================================================================= #

def split_into_subwords(text):
    merges = pkl.load(open('Data/Auxiliary/merge_ordered.pkl', 'rb'))
    subwords = []
    for word in text.split():
        for subword in merges:
            subword_count = word.count(subword)
            if subword_count > 0:
                word = word.replace(subword, ' ')
                subwords.extend([subword]*subword_count)
    return ' '.join(subwords)

# Testing
# print(split_into_subwords("hello my name is rocky"))


data_preprocessed_subwords = {k: [split_into_subwords(
    sentence) for sentence in v] for k, v in data_preprocessed.items()}

# Stats
# show_statistics(data_preprocessed_subwords)

data_train_subwords = []
for sentence in sentences_train:
    data_train_subwords.append(split_into_subwords(sentence))

data_val_subwords = []
for sentence in sentences_val:
    data_val_subwords.append(split_into_subwords(sentence))

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(data_train_subwords)
X_val = vectorizer.transform(data_val_subwords)

naive_classifier = MultinomialNB(alpha=1.25, fit_prior=False)
naive_classifier.fit(X_train, y_train)

predictions = naive_classifier.predict(X_val)

# ======================Performance Review =========================== #
# plot_confusion_matrix(y_val, predictions, ['Slovak', 'Czech', 'English'])
print(f1_score(y_val, predictions, average='weighted'))
# ==================================================================== #

# At this point the score we get from f1_score is 85%
