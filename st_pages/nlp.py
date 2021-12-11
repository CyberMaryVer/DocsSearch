import pandas as pd
import streamlit as st
import nltk
from fast_autocomplete import AutoComplete
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

nltk.download('punkt')


def get_words(df, column_name):
    words = {}
    for search in df[column_name].values:
        search = str(search).strip().lower()
        search = ' '.join(search.split()[:])
        words.update({search: {}})

    # words = {str(w).strip().lower():{} for w in df[column_name].values}
    return words


def get_corpus(df, column):
    lines = df[column].tolist()
    tokens = []
    exceptions_count = 0
    tokenizer = nltk.RegexpTokenizer('\w+')
    for line in lines:
        try:
            tokens += tokenizer.tokenize(line)
        except Exception as exc:
            exceptions_count += 1
            if exceptions_count == 1:
                print(exc)
    return [token.lower() for token in tokens]


def get_uni_freqs(corpus):
    fdist = nltk.FreqDist(corpus)
    freqs = []
    for k, v in fdist.items():
        freqs.append((k, v))
    return freqs


def get_bi_freqs(corpus):
    bgs = nltk.bigrams(corpus)
    fdist = nltk.FreqDist(bgs)
    freqs = []
    for k, v in fdist.items():
        freqs.append((k, v))
    return freqs


def get_tri_freqs(corpus):
    tgs = nltk.trigrams(corpus)
    fdist = nltk.FreqDist(tgs)
    freqs = []
    for k, v in fdist.items():
        freqs.append((k, v))
    return freqs


def get_nfreqs(corpus, n=4):
    sent = ' '.join(corpus)
    ngs = nltk.ngrams(sent.split(), n)

    fdist = nltk.FreqDist(ngs)
    freqs = []
    for k, v in fdist.items():
        freqs.append((k, v))
    return freqs


def bigram_trigram(df, column):
    corpus = get_corpus(df, column)

    uni_freqs = get_uni_freqs(corpus)
    bi_freqs = get_bi_freqs(corpus)
    tri_freqs = get_tri_freqs(corpus)
    four_freqs = get_nfreqs(corpus)
    five_freqs = get_nfreqs(corpus, n=5)
    six_freqs = get_nfreqs(corpus, n=6)

    return uni_freqs, bi_freqs, tri_freqs, four_freqs, five_freqs, six_freqs


def convert_ngrams(ngrams, format="dict"):
    converted_list = []
    converted_dict = {}
    for ngram in ngrams:
        key = ' '.join(ngram[0])
        key_r = ' '.join(ngram[0][::-1])
        converted_dict[key] = {}
        # converted_dict[key_r] = {}
        if key not in converted_list:
            converted_list.append(key)
            # converted_list.append(key_r)
    res = converted_dict if format == "dict" else converted_list
    return res


def get_searches(word, autocomplete, size=10):
    res = autocomplete.search(word=word, max_cost=3, size=size)
    searches = []
    for search in res:
        for subsearch in search:
            if subsearch not in searches:
                searches.append(subsearch)
    return searches


def preprocess_str(raw_str):
    res = raw_str.lower().strip()
    for x in "{}()<>/":
        res = res.replace(x, "")
    return res


def fuzz_search(x, search, thresh=57):
    if fuzz.partial_ratio(x, search) > thresh:
        return True
    return False


def get_projects(df, column, search, thresh):
    try:
        res = df.loc[df[column].apply(lambda x: search in preprocess_str(str(x)))]
        if res.shape[0] == 0:
            res = df.loc[df[column].apply(lambda x: fuzz_search(preprocess_str(str(x)), search, thresh))]
        return res
    except Exception as e:
        print(f"{e}")
    return


def format_text(text, search, color=(204, 34, 34)):
    formatted = []
    for word in text.split():
        if preprocess_str(word) in search:
            word = f'''<span style="background: rgb{color}; padding: 0.45em 0.6em; margin: 0px 0.25em; line-height: '
                f'1; border-radius: 0.35em;">{word}</span>'''
        formatted.append(word)
    res_text = " ".join(formatted)
    return res_text
