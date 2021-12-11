import pandas as pd
import streamlit as st

REESTR = "./../data/reestr.csv" if __name__ == "__main__" else "./data/reestr.csv"
REQUESTS = "./../data/requests.csv" if __name__ == "__main__" else "./data/requests.csv"

encoding_dict = {
    '10.Формирование ПМИ': 2,
    '11.Испытания': 2,
    '12.Подготовка ТЭО по результатам испытаний': 2,
    '13.Рассмотрение ФЗ результатов испытаний': 2,
    '14.Подготовка ТТ (ТЗ)': 2,
    '15.Заявка на финансирование': 2,
    '16.Адаптация решения для ОАО "РЖД"': 1,
    '17.Внедрение': 2,
    '2.Работа с проектом приостановлена': 0,
    '3.Назначен менеджер': 1,
    '4.Отправлено на доработку': 1,
    '5.Рассмотрение менеджером': 1,
    '7.Экспертиза': 1,
    '8.Рассмотрение ФЗ': 1,
    '9.Формирование ДК': 2
}


# print(__name__)

@st.experimental_memo
def get_dataframes():
    project_reestr = pd.read_csv(REESTR, skiprows=[0, 1, 3, 4], index_col=0)
    project_reestr.columns = ["project_name", "project_desc", "company_id", "company_representative",
                              "company_contacts",
                              "misc", "status_txt"]
    project_reestr["status"] = project_reestr.status_txt.map(encoding_dict)

    rzd_requests = pd.read_csv(REQUESTS, index_col=0, skiprows=[0, 2])
    rzd_requests.columns = ["year", "topic", "org", "q1", "q2", "q3", "q4"]
    rzd_requests.drop(labels=["q1", "q2", "q3", "q4"], axis=1, inplace=True)
    return project_reestr, rzd_requests
