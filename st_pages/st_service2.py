import pandas as pd
import streamlit as st
from fast_autocomplete import AutoComplete
from st_pages.nlp import bigram_trigram, convert_ngrams, get_searches, get_projects, format_text
from st_pages.dataframes import get_dataframes

project_reestr, rzd_requests = get_dataframes()


# @st.cache
def get_autocomplete(df=rzd_requests, column='topic'):
    ngrams = bigram_trigram(df, column)
    converted_ngrams = {}
    for ngr in ngrams:
        converted_ngrams.update(convert_ngrams(ngr))
    autocomplete = AutoComplete(words=converted_ngrams,
                                valid_chars_for_string='абвгдеёжзиклмнопрстуфхцчщшьыъэюя')
    return autocomplete


def service2():
    with st.form("Настройки быстрого поиска"):
        ac_choice = st.radio("Выберите данные для обработки", ("Открытые запросы", "Проекты в базе"))
        st.form_submit_button("Submit")

    if ac_choice == "Открытые запросы":
        data = rzd_requests
        col = "topic"
    else:
        data = project_reestr
        col = "project_name"

    autocomplete = get_autocomplete(df=data, column=col)
    st.markdown("## Демонстрация функционала быстрого поиска")
    user_input = st.text_input("Введите неполный запрос:")

    user_output = get_searches(user_input, autocomplete=autocomplete)
    for s in user_output:
        st.write(f"* {s}")

    if len(user_output) > 0:
        user_choice = st.selectbox("Выбор результата", user_output)
        res = get_projects(data, col, user_choice, thresh=72)

        st.dataframe(res)
        st.markdown(format_text(res[col].values[0], user_choice, color=(119, 189, 239)), unsafe_allow_html=True)