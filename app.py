import streamlit as st
from streamlit import StreamlitAPIException
from st_pages.st_utils import st_img, st_title
from st_pages.st_service2 import service2
from st_pages.st_service1 import st_serious_search
from st_pages.st_analysis import st_load_docx
from st_pages.get_embeddings import get_them_all

# set page settings
try:
    st.set_page_config(page_title="web-app", page_icon=":bar_chart:", layout="wide",
                       menu_items={
                           'Get Help': 'https://www.rferl.org/a/kamchatka-volcanoes-ballistic-missile/31358301.html',
                           'About': "### Undefined Variable\n----\nNLP Jazz"
                       })
except StreamlitAPIException as e:
    st.legacy_caching.clear_cache()
    print(e)


def main():
    side_menu_list = [
        "Анализ документа",
        "Векторный поиск - демо",
        "Быстрый поиск - демо",
        "Информация о проекте"
    ]

    side_menu_choice = st.sidebar.selectbox("", side_menu_list, key="side_menu")
    side_menu_idx = side_menu_list.index(side_menu_choice)
    # st_img("./images/logo.jfif", sidebar=True, width=300)

    # username = st.session_state["username"]
    if side_menu_idx == 0:
        st_title("Семантический анализ документа")
        st_load_docx()

    elif side_menu_idx == 1:
        st_title("Векторный поиск - демо")
        st_serious_search()

    elif side_menu_idx == 2:
        st_title("Быстрый поиск - демо")
        service2()

    elif side_menu_idx == 3:
        st_title("Информация о проекте")
        pass


if __name__ == "__main__":
    get_them_all()
    main()