import pandas as pd
import streamlit as st
from fast_autocomplete import AutoComplete
from st_pages.nlp import bigram_trigram, convert_ngrams, get_searches, get_projects, show_project
from st_pages.dataframes import get_dataframes
from st_pages.st_utils import save_logs

VALID_CHARS = "абвгдеёжзиклмнопрстуфхцчщшьыъэюяabcdefghijklmnopqrstuvwxyz"
project_reestr, rzd_requests = get_dataframes()


@st.experimental_singleton
def get_autocomplete(df, columns):
    converted_ngrams = {}
    for col in columns:
        ngrams = bigram_trigram(df, col)
        for ngr in ngrams:
            converted_ngrams.update(convert_ngrams(ngr))
    autocomplete = AutoComplete(words=converted_ngrams,
                                valid_chars_for_string=VALID_CHARS)
    return autocomplete


def service2():
    with st.form("Настройки быстрого поиска"):
        ac_choice = st.radio("Выберите данные для обработки", ("Открытые запросы", "Проекты в базе"))
        st.form_submit_button("Submit")

    if ac_choice == "Открытые запросы":
        data = rzd_requests
        columns = ["topic",]
    else:
        data = project_reestr
        columns = ["project_name", "project_desc"]

    is_show = st.sidebar.checkbox("Отображать необработанные результаты")
    autocomplete = get_autocomplete(df=data, columns=columns)
    st.markdown("## Демонстрация функционала быстрого поиска")
    st.markdown("----")
    st.markdown("** *Введите неполный запрос и нажмите на Enter, чтобы увидеть список подсказок:* **")
    user_input = st.text_input("Введите неполный запрос:")
    save_logs(f"FAST SEARCH: {user_input} ({ac_choice})")

    user_output = get_searches(user_input, autocomplete=autocomplete)
    for s in user_output:
        s_ = ' '.join([w for w in s.split() if len(w) > 1])
        s = s_ if len(s_) > 0 else s
        st.write(f"* {s}")

    placeholder = st.empty()
    df_res = pd.DataFrame()

    if len(user_output) > 0:
        if is_show:
            for output_string in user_output:
                res_ = get_projects(data, columns, output_string, thresh=72)
                df_res = pd.concat((df_res, res_))
            with placeholder:
                df_res = df_res.sort_values(by=['status'], ascending=False) \
                    if ac_choice == "Проекты в базе" else df_res
                st.dataframe(df_res)

        st.markdown("----")
        st.markdown("** *Пример выдачи при выборе результата из подсказок: * **")
        user_choice = st.selectbox("Выбор результата", user_output)
        save_logs(f"FAST SEARCH [NEXT]: {user_input} ({ac_choice})")
        res = get_projects(data, columns, user_choice, thresh=72)

        st.markdown("----")
        st.dataframe(res)

        for col in columns[:1]:
            # st.markdown(format_text(res[col].values[0], user_choice, color=(119, 189, 239)), unsafe_allow_html=True)
            for val in res[col].values:
                show_project(user_input, user_choice, val, data, col)
