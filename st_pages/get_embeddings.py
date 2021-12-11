import requests
import subprocess
import os

EMBEDDINGS = "./../data/embeddings.pk" if __name__ == "__main__" else "./data/embeddings.pk"


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_them_all():
    if not os.path.exists(EMBEDDINGS):
        download_file_from_google_drive("1Fjk6zk6qikPwv8ihjHP4LtZSExVfSD4h", EMBEDDINGS)
    # subprocess.run("python -m spacy download ru_core_news_md")
    print("Embeddings are here")


if __name__ == "__main__":
    get_them_all()