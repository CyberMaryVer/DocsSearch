import pickle as pk
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from st_pages.dataframes import get_dataframes
from st_pages.nlp import get_nfreqs, get_uni_freqs, join_freqs

CONVERTED_REESTR = "./../data/converted_reestr.pk" if __name__ == "__main__" else "./data/converted_reestr.pk"
BERT_EMBEDDINGS = "./../data/embeddings.pk" if __name__ == "__main__" else "./data/embeddings.pk"
LONG_BERT_EMBEDDINGS = "./../data/lembeddings.pk" if __name__ == "__main__" else "./data/lembeddings.pk"
TFDF_EMBEDDINGS = "./../data/tfidf.pk" if __name__ == "__main__" else "./data/tfidf.pk"
TFDF_VECTORIZER = "./../data/tfidf_vectorizer.pk" if __name__ == "__main__" else "./data/tfidf_vectorizer.pk"


# set BERT
# @st.cache(allow_output_mutation=True)
@st.experimental_singleton
def get_bert():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return model


@st.cache(allow_output_mutation=True)
def get_embeddings():
    with open(BERT_EMBEDDINGS, 'rb') as fp:
        embeddings_distilbert = pk.load(fp)

    with open(LONG_BERT_EMBEDDINGS, 'rb') as fp:
        embeddings_long = pk.load(fp)

    with open(CONVERTED_REESTR, 'rb') as fp:
        converted_reestr = pk.load(fp)

    with open(TFDF_EMBEDDINGS, 'rb') as fp:
        vectors_tfidf = pk.load(fp)

    with open(TFDF_VECTORIZER, 'rb') as fp:
        tfidf_vectorizer = pk.load(fp)

    return embeddings_distilbert, embeddings_long, converted_reestr, vectors_tfidf, tfidf_vectorizer


def get_df_by_indexes(indexes):
    reestr, _ = get_dataframes()
    search_result = reestr[reestr.index.isin(indexes)].sort_values(by=['status'], ascending=False)
    return search_result


def update_indexes(df, all_indexes):
    new_indexes = list(df.index)
    for idx in new_indexes:
        if idx not in all_indexes:
            all_indexes.append(idx)
    return all_indexes


def find_indexes_in_converted_reestr(search_request):
    from fuzzywuzzy import fuzz
    embeddings_distilbert, embeddings_long, converted_reestr, vectors_tfidf, tfidf_vectorizer = get_embeddings()

    all_indexes = []
    for idx, sent in enumerate(converted_reestr):
        ratio = fuzz.partial_ratio(sent, search_request)
        if ratio > 80 and idx not in all_indexes:
            all_indexes.append(idx)

    st.write({x[0]: x[1] for x in enumerate(converted_reestr) if x[0] in all_indexes})
    return all_indexes


def find_similar(vector_representation, all_representations, k=1):
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        # print(min(similarities))
        return [np.argmax(similarities)]
    elif k is not None:
        # print(np.sort(similarities)[-k:])
        return np.flip(similarities.argsort()[-k:][::1])


def preprocess_str(raw_str):
    res = raw_str.lower().strip()
    for x in "{}()<>/":
        res = res.replace(x, "")
    return res


def fuzz_search(x, search, thresh=57):
    if fuzz.partial_ratio(x, search) > thresh:
        return True
    return False


def get_projects(df, columns, search, thresh):
    try:
        all_res = pd.DataFrame()
        for col in columns:
            res = df.loc[df[col].apply(lambda x: search in preprocess_str(str(x)))]
            if res.shape[0] == 0:
                res = df.loc[df[col].apply(lambda x: fuzz_search(preprocess_str(str(x)), search, thresh))]
            all_res = pd.concat((all_res, res))
        all_res.drop_duplicates(subset=columns, inplace=True, keep='last')
        return all_res
    except Exception as e:
        print(f"{e}")
    return


def find_projects_by_request(search_request):
    project_reestr, rzd_requests = get_dataframes()
    embeddings_distilbert, embeddings_long, converted_reestr, vectors_tfidf, tfidf_vectorizer = get_embeddings()
    model = get_bert()

    # set TFIDF
    lemmatizer = WordNetLemmatizer()

    REPRESENTATIONS = {
        "tfidf": vectors_tfidf,
        "bert": embeddings_distilbert
    }
    idxs = find_sentence_idxs(search_request,
                              representations=REPRESENTATIONS,
                              token_fmt="tfidf",
                              bert_model=model,
                              embeddings_distilbert_long=embeddings_long,
                              tfidf_vectorizer=tfidf_vectorizer,
                              tfidf_lemmatizer=lemmatizer)
    searches = np.array(converted_reestr)[idxs]

    all_indexes = []
    if searches.size > 0:
        for s in searches:
            res = get_projects(project_reestr, ["project_desc"], s, 64)
            res_indexes1 = list(res.head(4).index) if res is not None else []
            vect = model.encode([s])
            similar_indexes = find_similar(vect, embeddings_long)
            similar_data = project_reestr.iloc[similar_indexes].sort_values(by=['status'])
            res_indexes2 = list(similar_data.index)

            for idxs in [res_indexes1, res_indexes2]:
                for idx in idxs:
                    if idx not in all_indexes:
                        all_indexes.append(idx)

        print(all_indexes)
        search_result = project_reestr[project_reestr.index.isin(all_indexes)].sort_values(by=['status'],
                                                                                           ascending=False)
        return search_result


def find_sentence_idxs(sent,
                       representations,
                       token_fmt,
                       bert_model,
                       embeddings_distilbert_long,
                       tfidf_vectorizer,
                       tfidf_lemmatizer,
                       ):
    res = []
    for i in [9, 8, 7, 6, 5, 4, 3]:
        idxs = []
        try:
            n_freqs = get_nfreqs([sent], n=i)
            converted = join_freqs(n_freqs)

            for ngm in converted:
                if token_fmt == "tfidf":
                    vect = tfidf_vectorizer.transform(
                        [" ".join([tfidf_lemmatizer.lemmatize(token.lower()) for token in word_tokenize(ngm)])])
                elif token_fmt == "bert":
                    vect = bert_model.encode([ngm])
                similar_indexes = find_similar(vect, representations[token_fmt])
                for idx in similar_indexes:
                    res.append(idx)

            idxs = [int(x[0]) for x in get_uni_freqs([str(f) for f in res])]
            if len(idxs) > 1:
                return idxs
        except Exception as e:
            # print(f"{type(e)}: {e}")
            pass
    if token_fmt == "bert":
        idxs = find_similar(vect, embeddings_distilbert_long)
    if len(idxs) > 0:
        return idxs
    print("not found")
    return