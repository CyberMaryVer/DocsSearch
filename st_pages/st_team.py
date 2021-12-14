import streamlit as st
from st_pages.st_utils import st_img


def st_team():
    st.title("Team")
    st_img("./images/team.png", width=800)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.info("Мария - NLP инженер")
        st.info("Алина - дизайнер")
        st.info("Максим - дата-аналитик")
        st.info("Павел - дата-аналитик")
    with col2:
        # st_img("./data/tg.png")
        st.markdown(f""":woman: [Мария](https://t.me/cybermary)""",
                    unsafe_allow_html=True)
        st.markdown(f""":woman: [Алина](https://t.me/alinka27_official)""",
                    unsafe_allow_html=True)
        st.markdown(f""":man: [Максим](https://t.me/Makcimilian)""",
                    unsafe_allow_html=True)
        st.markdown(f""":man: [Павел](https://t.me/pavel_grom_gq)""",
                    unsafe_allow_html=True)
        with st.form():
            login = st.text_input("Login")
            password = st.text_input("Password")
            submit = st.form_submit_button("Submit")

        if submit:
            is_login = login == st.secrets["USER"]
            is_password = password == st.secrets["PASSWORD"]
            if is_password and is_login:
                with open("log.txt", "r") as reader:
                    logs = reader.read()
                st.write(logs)


