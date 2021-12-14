import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from st_pages.vis_prompt import SysColors

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('russian')
stem = SnowballStemmer('russian')
punc = RegexpTokenizer(r'\w+')

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


def join_freqs(freqs):
    txt = [[y.lower() for y in x[0] if len(y) > 1] for x in freqs]
    txt = [" ".join(x) for x in txt]
    return txt


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


def preprocess_raw(raw_str, lemmatize=True):
    curr_raw = word_tokenize(raw_str, 'russian')
    if not lemmatize:
        return curr_raw
    new_raw = " ".join([stem.stem(i) for i in curr_raw if i not in stop_words and i.isalpha()])
    return new_raw


def fuzz_search(x, search, thresh=64, lemmatize=True):
    if lemmatize:
        x = preprocess_raw(x)
        search = preprocess_raw(search)
    if fuzz.partial_ratio(x, search) > thresh:
        return True
    return False


def get_projects(df, columns, search, thresh):
    try:
        all_res = pd.DataFrame()

        for col in columns:
            # search for exact match
            res = df.loc[df[col].apply(lambda x: search in preprocess_str(str(x)))]

            if not res.shape[0] == 0:
                all_res = pd.concat((all_res, res))
            else:
                print(SysColors.WARNING, "NO EXACT MATCHES", SysColors.RESET)

            # search for Levenshtein closest match
            res = df.loc[df[col].apply(lambda x: fuzz_search(preprocess_str(str(x)), search, thresh, True))]

            all_res = pd.concat((all_res, res))

        # filter only unique results
        all_res.drop_duplicates(subset=columns, inplace=True, keep='last')
        return all_res

    except Exception as e:
        print(f"{e}")
    return


def format_text(text, search, color=(204, 34, 34)):
    formatted = []
    text = text if type(text) is str else ""
    for word in text.split():
        if len(word) > 3 and preprocess_str(word) in search:
            word = f'''<span style="background: rgb{color}; padding: 0.4em 0.4em; margin: 0px 0.06em; line-height: '
                f'1; border-radius: 0.35em;">{word}</span>'''
        elif len(word) > 1 and preprocess_str(word) in search:
            word = f'''<span style="background: rgb(242, 242, 242); padding: 0.4em 0.4em; margin: 0px 0.06em; 
            line-height:1; border-radius: 0.35em;">{word}</span>'''
        formatted.append(word)
    res_text = " ".join(formatted)
    return res_text


def show_project(user_input, search, val, df, col, name_col=None,
                 color=(242, 205, 205), second_color=(242, 205, 205)):
    name_col = col if name_col is None else name_col
    project_name = df.loc[df[col] == val][name_col].to_list()
    if len(project_name) > 0:
        st.code(project_name)
        html = format_text(val, user_input, color=second_color)
        html = format_text(html, search, color=color)
        st.markdown(html, unsafe_allow_html=True)
