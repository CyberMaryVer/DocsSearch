import streamlit as st
import spacy

nlp = spacy.load('ru_core_news_md')
ENTITIES = ["ORG", "GPE"]


def format_red_text(text, url=""):
    return f'''<span style="background: rgb(204,34,34); padding: 0.24em 0.24em; margin: 0px 0.2em; line-height: '
                f'1; border-radius: 0.35em;">{text}</span>'''


def entity_info(text, eob, color=(242, 242, 242), contour=(183, 183, 183)):
    ent_info = f'''<b style="font-size:8px">    {eob}</b>''' if eob else ""
    return f'''<span style="background: rgb{color}; padding: 0.2em 0.2em; margin: 0px 0.25em; line-height: '
                f'1; border-radius: 0.35em; border: 2px solid rgb{contour};">{text}{ent_info}</span>'''


def text2tokens(text, ents):
    doc = nlp(text)
    space = " "
    html = ""
    for token in doc:

        if token.ent_type_ == "DATE" or token.ent_type_ == "MONEY":
            html += space + entity_info(token.text, False, color=(255, 211, 211), contour=(255, 42, 42))
        elif token.tag_ == "PRP" or token.pos == "NUM":
            html += space + entity_info(token.text, False, color=(167, 234, 12))
        elif token.ent_type_ == "PER":
            html += space + entity_info(token.text, False, color=(124, 226, 161), contour=(0, 255, 0))
        elif token.ent_type_ in ents:
            html += space + entity_info(token.text, False)
        elif token.pos == "PUNCT":
            html += token.text
        else:
            html += space + token.text  # + (space+token.tag_)

    return html


def format_string_as_spacy(text):
    st_text = text2tokens(text, ents=ENTITIES)
    st.markdown(st_text, unsafe_allow_html=True)


# @st.cache
def format_for_streamlit(text, text_file=None):
    if text is None and text_file is not None:
        with open(text_file, "r") as reader:
            text = nlp(reader.read())
    elif text is None:
        return
    format_string_as_spacy(text)
