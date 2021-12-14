import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from st_pages.wordcloud import create_ngrams
from st_pages.search_engine import find_sentence_idxs, get_projects, \
    get_bert, get_embeddings, get_dataframes, find_similar


def st_load_docx_and_analyze():
    from st_pages.st_analysis import preproc_raw
    from docx import Document

    doc_file = st.file_uploader("Загрузите документ для анализа", type=["docx"])

    if doc_file is not None:
        content = doc_file.read()
        with open("tmp.docx", "wb+") as writer:
            writer.write(content)
        doc = Document("tmp.docx")
        st.write(type(doc))
        # st.write("WTF")
        requirements_txt = []
        requirements_raw = []

        try:
            for para in doc.paragraphs:
                if "требован" in para.text:
                    raw_txt = [t for t in para.text.split("Преимущества участия")[0].split("\n") if t != ""]
                    cleaned_txt = preproc_raw(para.text.replace("\t", " ").strip().lower(),
                                              lemmatize=False)
                    if not cleaned_txt == "":
                        requirements_txt.append(cleaned_txt)
                        requirements_raw.append(raw_txt)
            # for r in requirements_txt:
            # ngs = create_ngrams(doc_file.name.split(".")[0].strip().split(), 6)
            # sents = [n for n in ngs]
            # st.write(sents)
            request = doc_file.name.split(".")[-2].strip()
            # st.write({"search text": request})
            search_res = find_projects_by_request(doc_file.name.split(".")[0].strip())
            if search_res is not None:
                st.text("Похожий проект:")
                st.dataframe(search_res)
                all_indexes = list(search_res.index)
            else:
                st.text("Похожих проектов не найдено")
                all_indexes = []

            if st.button("Найти частично похожие проекты"):
                st.text("Проводим поиск по ближайшим...")
                placeholder = st.empty()
                ngs = create_ngrams(request.split(), 6)
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


        except Exception as e:
            st.write("Проверьте формат файла")
            print(f"{type(e)}: {e}")


@st.cache
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
    embeddings_distilbert, emdeddings_long, converted_reestr, vectors_tfidf, tfidf_vectorizer = get_embeddings()

    all_indexes = []
    for idx, sent in enumerate(converted_reestr):
        ratio = fuzz.partial_ratio(sent, search_request)
        if ratio > 80 and idx not in all_indexes:
            all_indexes.append(idx)

    st.write({x[0]: x[1] for x in enumerate(converted_reestr) if x[0] in all_indexes})
    return all_indexes


@st.cache
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
