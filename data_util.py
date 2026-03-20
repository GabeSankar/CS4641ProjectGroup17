from datasets import load_dataset
import nltk
from collections import Counter
import numpy as np
import re
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
import string
import pandas as pd
from scipy.sparse import lil_matrix

def pull_data_to_pd(checkpoint):
    dataset = load_dataset(checkpoint)
    return pd.DataFrame(dataset)

def tokenize(text):
    '''
    Tokanization
    '''
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())
    return sentences, words

def get_ngrams(text, n=2):
    """
    get list of n grams, which is in the form of tuples
    """
    _, words = tokenize(text)
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def build_ngram_vocab(texts, n=2, min_freq=2):
    '''build n gram ratio vocabulary for entire corpus with a min freq'''
    counter = Counter()
    for text in texts:
        ngrams = get_ngrams(text, n)
        counter.update(ngrams)

    #min lenght filtering 
    filtered_ngrams = [ng for ng, c in counter.items() if c >= min_freq]

    vocab = {ng: i for i, ng in enumerate(filtered_ngrams)}
    return vocab

def ngram_ratio_vector(text, vocab, n=2):
    '''Get vector for ngram proportions, will be very sparse'''
    ngrams = get_ngrams(text, n)
    counts = Counter(ngrams)
    total = len(ngrams)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for ng, count in counts.items():
        if ng in vocab:
            vec[vocab[ng]] = count / total if total else 0
    return vec

def ngram_ratio_vector_sparse(text, vocab, n=2):
    ngrams = get_ngrams(text, n)
    counts = Counter(ngrams)
    total = len(ngrams)
    
    vec = lil_matrix((1, len(vocab)), dtype=np.float32)
    for ng, count in counts.items():
        if ng in vocab:
            vec[0, vocab[ng]] = count / total if total else 0
    return vec

def build_stylometric_vector(text, include_pos=True):
    """
    Make stylometric feature vector, also already normalized by text length
    """
    sentences, words = tokenize(text)

    # simple counts
    num_words = len(words)
    num_chars = len(text)

    # sentence length (doesnt grow with document text)
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    var_sent_len = np.var(sent_lengths) if sent_lengths else 0

    # word lengths
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0

    # token type ration (unique words / number of words)
    ttr = len(set(words)) / num_words if num_words else 0


    # punctiation counter
    punct_counts = Counter(c for c in text if c in string.punctuation)
    punct_total = sum(punct_counts.values())
    punct_ratio = punct_total / num_chars if num_chars else 0

    # punctuation features organized by ['.', ',', '!', '?', ';', ':']
    punct_features = [punct_counts[p] for p in ['.', ',', '!', '?', ';', ':']]

    #set puncuation features as ratios
    punct_features = [
    punct_counts[p] / punct_total if punct_total else 0
    for p in ['.', ',', '!', '?', ';', ':']
    ]
    
    # POS features are ratios
    pos_features = []
    if include_pos and words:
        tagged = nltk.pos_tag(words)
        tag_counts = Counter(tag for _, tag in tagged)
        total_tags = sum(tag_counts.values())

        #POS features organized by Noun Verb Adjetive Adverb Determiner Preposition and Personal pronoun
        tag_set = ['NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'PRP']
        pos_features = [tag_counts[tag] / total_tags for tag in tag_set]

    # vector with flattened dim
    vector = [
        avg_sent_len,
        var_sent_len,
        avg_word_len,
        ttr,
        punct_ratio,
        *punct_features,
        *pos_features
    ]

    return np.array(vector, dtype=float)

def flip_dataframe(df):
    '''
    Method to flip dataframe columns into long format and remove any missing text
    '''
    #remove prompt column before flipping lengthwise
    df_flipped = df.drop(columns=['prompt'])
    df_flipped = df_flipped.melt(var_name='label', value_name='text')

    df_flipped = df_flipped.dropna(subset=['text'])
    df_flipped = df_flipped[df_flipped['text'].str.strip() != '']

    return df_flipped
