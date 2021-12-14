import streamlit as st
import pandas as pd
import base64
from PIL import Image
import streamlit.components.v1 as components
from datetime import datetime
import qrcode


def st_title(title_text, color=(138, 139, 148)):
    st.markdown(f"""<p style="background: rgb{color}; padding: 1.55em 1.55em; margin: 0px 0.0em; line-height: '
                    f'1; border-radius: 0.25em;"><b>{title_text.upper()}</b></p>""", unsafe_allow_html=True)
    st.markdown("----")


def st_html(html_path, width=600, height=400, scrolling=False):
    html = open(html_path, 'r', encoding='utf-8')
    source_code = html.read()
    components.html(source_code, width, height, scrolling=scrolling)


def st_iframe(url, width=800, height=400, scrolling=False):
    components.iframe(url, width=width, height=height, scrolling=scrolling)


def st_gif(gif_path, sidebar=False):
    code = _generate_base64_str_for_gif(gif_paths=gif_path)
    if sidebar:
        st.sidebar.markdown(code, unsafe_allow_html=True)
    else:
        st.markdown(code, unsafe_allow_html=True)


def st_img(img_path, sidebar=False, width=600):
    img_to_show = Image.open(img_path)
    if sidebar:
        st.sidebar.image(img_to_show, width=width)
    else:
        st.image(img_to_show, width=width)


def st_imgs(img_paths, width=600):
    imgs_to_show = [Image.open(img_path) for img_path in img_paths]
    st.image(imgs_to_show, width=width)


def _generate_base64_str_for_gif(gif_bytes=None, gif_paths=None):
    if gif_paths is None and gif_bytes is None:
        raise SyntaxError("gif_bytes or gif_paths should be defined")

    if gif_paths:
        try:
            gif_bytes = _load_bytes(gif_paths)
        except FileNotFoundError:
            return

    data_url = base64.b64encode(gif_bytes[0]).decode("utf-8")
    return f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">'


def _load_bytes(paths):
    bytes = []

    if type(paths) is str:
        with open(paths, "rb") as reader:
            bytes.append(reader.read())

    elif type(paths) is list:
        for p in paths:
            with open(p, "rb") as reader:
                bytes.append(reader.read())

    else:
        raise TypeError("wrong type of data")

    return bytes


def create_qrcode(url, out_file="qr.png", box_size=6, border=4):
    qr = qrcode.QRCode(version=1, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(out_file)
    return img


def format_text(text, color=(226, 26, 26)):
    return f'''<span style="background: rgb{color}; padding: 0.4em 0.4em; margin: 0px 0.25em; line-height:1; 
                border-radius: 0.15em;">{text}</span>'''


def color_text(text, color=(226, 26, 26)):
    return f'''<span style="color: rgb{color}; padding: 0.45em 0.6em; margin: 0px 0.25em; line-height:1; 
                border-radius: 0.15em;">{text}</span>'''


@st.experimental_memo
def load_df(df_path):
    df = pd.read_csv(df_path, index_col=0)
    return df


def save_logs(txt):
    with open("log.txt", "a", encoding="utf-8") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


if __name__ == "__main__":
    pass
