import nltk
import pickle as pk
import streamlit as st
import numpy as np
# import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
# from gensim.models.doc2vec import Doc2Vec
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from st_pages.dataframes import get_dataframes
from st_pages.nlp import get_nfreqs, get_uni_freqs, get_projects, format_text

nltk.download('punkt')
nltk.download('wordnet')
# Number of top
K = 5
CONVERTED_REESTR = "./../data/converted_reestr.pk" if __name__ == "__main__" else "./data/converted_reestr.pk"
BERT_EMBEDDINGS = "./../data/embeddings.pk" if __name__ == "__main__" else "./data/embeddings.pk"
TFDF_EMBEDDINGS = "./../data/tfidf.pk" if __name__ == "__main__" else "./data/tfidf.pk"
TFDF_VECTORIZER = "./../data/tfidf_vectorizer.pk" if __name__ == "__main__" else "./data/tfidf_vectorizer.pk"


# set BERT
@st.cache(allow_output_mutation=True)
def get_bert():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return model


@st.cache(allow_output_mutation=True)
def get_embeddings():
    with open(BERT_EMBEDDINGS, 'rb') as fp:
        embeddings_distilbert = pk.load(fp)

    with open(CONVERTED_REESTR, 'rb') as fp:
        converted_reestr = pk.load(fp)

    with open(TFDF_EMBEDDINGS, 'rb') as fp:
        vectors_tfidf = pk.load(fp)

    with open(TFDF_VECTORIZER, 'rb') as fp:
        tfidf_vectorizer = pk.load(fp)

    return embeddings_distilbert, converted_reestr, vectors_tfidf, tfidf_vectorizer


def get_long_embeddings(data, model):
    def clean(s):
        return str(s).lower().strip().replace("(", "").replace(")", "").replace("/", "")

    embeddings_long = model.encode([clean(x) for x in data])
    return embeddings_long


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


def join_freqs(freqs):
    txt = [[y.lower() for y in x[0]] for x in freqs]
    txt = [" ".join(x) for x in txt]
    return txt


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


def st_serious_search():
    # st.legacy_caching.clear_cache()

    project_reestr, rzd_requests = get_dataframes()
    embeddings_distilbert, converted_reestr, vectors_tfidf, tfidf_vectorizer = get_embeddings()
    model = get_bert()
    embeddings_distilbert_long = get_long_embeddings(project_reestr["project_desc"].values, model)

    # set TFIDF
    lemmatizer = WordNetLemmatizer()

    REPRESENTATIONS = {
        "tfidf": vectors_tfidf,
        "bert": embeddings_distilbert
    }

    st.markdown("## Демонстрация функционала векторного поиска")
    user_input = st.text_input("Введите запрос:",
                               value="Беспилотное управление локомотивом с применением технологий машинного зрения")
    approach = st.selectbox("Выберите модель", ("tfidf",))

    st.markdown(f"### Текст запроса:")
    st.markdown(user_input)

    st.markdown(f"### Выбранная модель:")
    st.markdown(approach)

    if True:
        print("\n", "-" * 100)
        st.write(f"### Результат:")
        idxs = find_sentence_idxs(user_input,
                                  representations=REPRESENTATIONS,
                                  token_fmt=approach,
                                  bert_model=model,
                                  embeddings_distilbert_long=embeddings_distilbert_long,
                                  tfidf_vectorizer=tfidf_vectorizer,
                                  tfidf_lemmatizer=lemmatizer)
        st.write(f"**Индексы ngrams**: {idxs}")
        searches = np.array(converted_reestr)[idxs]
        st.write(f"**Ngrams**: {searches}")
        all_indexes = []
        if searches.size > 0:
            st.markdown(f"*Поиск по ближайшим ngrams: {searches}*")
            for s in searches:
                # print(s)
                st.markdown(f"----")
                st.markdown(f"---- ngram: {s} ----")
                st.markdown("#### Ближайшие объекты:")

                res = get_projects(project_reestr, "project_desc", s, 64)
                res_indexes1 = list(res.head(4).index)
                vect = model.encode([s])
                similar_indexes = find_similar(vect, embeddings_distilbert_long)
                similar_data = project_reestr.iloc[similar_indexes].sort_values(by=['status'])
                st.dataframe(similar_data)
                res_indexes2 = list(similar_data.index)

                for val in set(project_reestr.iloc[similar_indexes].project_desc.values.tolist()):
                    st.code(val)
                    st.markdown(format_text(val, s, color=(119, 189, 239)), unsafe_allow_html=True)

                if res is not None and res.shape[0] >= 1:
                    st.dataframe(res.head(4).sort_values(by=['status']))
                    for val in set(res.project_desc.values.tolist()[:4]):
                        st.code(val)
                        st.markdown(format_text(val, s, color=(119, 189, 239)), unsafe_allow_html=True)

                for idxs in [res_indexes1, res_indexes2]:
                    for idx in idxs:
                        if idx not in all_indexes:
                            all_indexes.append(idx)

            print(all_indexes)
            st.sidebar.text("Результаты поиска")
            st.sidebar.dataframe(project_reestr[project_reestr.index.isin(all_indexes)].sort_values(by=['status'],
                                                                                                    ascending=False))
