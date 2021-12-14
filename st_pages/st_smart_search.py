import pickle as pk

import nltk
import numpy as np
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

from st_pages.dataframes import get_dataframes
from st_pages.nlp import get_projects, format_text, show_project
from st_pages.search_engine import find_indexes_in_converted_reestr, update_indexes, \
    get_bert, get_embeddings, get_df_by_indexes, find_projects_by_request, find_sentence_idxs, find_similar
from st_pages.wordcloud import create_ngrams
from st_pages.st_utils import save_logs

nltk.download('punkt')
nltk.download('wordnet')

K = 5
CONVERTED_REESTR = "./../data/converted_reestr.pk" if __name__ == "__main__" else "./data/converted_reestr.pk"
BERT_EMBEDDINGS = "./../data/embeddings.pk" if __name__ == "__main__" else "./data/embeddings.pk"
LONG_BERT_EMBEDDINGS = "./../data/lembeddings.pk" if __name__ == "__main__" else "./data/lembeddings.pk"
TFDF_EMBEDDINGS = "./../data/tfidf.pk" if __name__ == "__main__" else "./data/tfidf.pk"
TFDF_VECTORIZER = "./../data/tfidf_vectorizer.pk" if __name__ == "__main__" else "./data/tfidf_vectorizer.pk"


def st_serious_search():

    project_reestr, rzd_requests = get_dataframes()
    embeddings_distilbert, embeddings_long, converted_reestr, vectors_tfidf, tfidf_vectorizer = get_embeddings()
    model = get_bert()

    # set TFIDF
    lemmatizer = WordNetLemmatizer()

    REPRESENTATIONS = {
        "tfidf": vectors_tfidf,
        "bert": embeddings_distilbert
    }

    st.markdown("## Демонстрация функционала умного поиска")
    user_input = st.text_input("Введите запрос:",
                               value="Беспилотное управление локомотивом с применением технологий машинного зрения")
    approach = "tfidf"  # st.selectbox("Выберите модель", ("tfidf",))

    if True:
        save_logs(f"SMART SEARCH: {user_input}")
        st.markdown("----")
        st.write(f"### Результат:")
        idxs = find_sentence_idxs(user_input,
                                  representations=REPRESENTATIONS,
                                  token_fmt=approach,
                                  bert_model=model,
                                  embeddings_distilbert_long=embeddings_long,
                                  tfidf_vectorizer=tfidf_vectorizer,
                                  tfidf_lemmatizer=lemmatizer)

        if idxs is None:
            st.text("[Запрос слишком короткий - в данном случае в программе используется быстрый поиск]")
        else:
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

                    res = get_projects(project_reestr, ["project_desc", "project_name"], s, 64)

                    if res is not None:
                        # st.write("res is not none")
                        res_indexes1 = list(res.head(4).index)
                        vect = model.encode([s])
                        similar_indexes = find_similar(vect, embeddings_long)
                        similar_data = project_reestr.iloc[similar_indexes].sort_values(by=['status'])
                        # st.dataframe(similar_data)
                        res_indexes2 = list(similar_data.index)

                        st.markdown("**Похожий проект**")
                        for val in set(project_reestr.iloc[similar_indexes].project_desc.values.tolist()):
                            show_project(user_input, s, val, project_reestr, col="project_desc",
                                         name_col="project_name")

                        if res is not None and res.shape[0] >= 1:
                            st.write("----")
                            st.markdown("**Еще похожие проекты**")
                            # st.dataframe(res.head(4).sort_values(by=['status']))
                            for val in set(res.project_desc.values.tolist()[:4]):
                                show_project(user_input, s, val, project_reestr, col="project_desc",
                                         name_col="project_name")

                        for idxs in [res_indexes1, res_indexes2]:
                            for idx in idxs:
                                if idx not in all_indexes:
                                    all_indexes.append(idx)
                    else:
                        # st.write("res is none")
                        placeholder = st.empty()

                        with st.spinner("Проводим поиск по ближайшим..."):
                            ngs = create_ngrams(user_input.split(), 6)
                            st.write(ngs)
                            for idx, ngm in enumerate(ngs):
                                if idx % 3 == 0:  # iteration interval
                                    find_indexes_in_converted_reestr(ngm)
                                    st.write({"search text": ngm})
                                    search_res = find_projects_by_request(ngm)
                                    if search_res is not None:
                                        all_indexes = update_indexes(search_res, all_indexes)
                                with placeholder:
                                    show_df = get_df_by_indexes(all_indexes)
                                    st.dataframe(show_df)

                print(all_indexes)
                st.sidebar.text("Результаты поиска")
                st.sidebar.dataframe(project_reestr[project_reestr.index.isin(all_indexes)].sort_values(by=['status'],
                                                                                                        ascending=False))
