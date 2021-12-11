from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import wordcloud

# plt.rcParams["figure.figsize"] = [16, 9]

def create_ngrams(token_list, nb_elements):
    """
    Create n-grams for list of tokens
    Parameters
    ----------
    token_list : list
        list of strings
    nb_elements :
        number of elements in the n-gram
    Returns
    -------
    Generator
        generator of all n-grams
    """
    ngrams = zip(*[token_list[index_token:] for index_token in range(nb_elements)])
    return (" ".join(ngram) for ngram in ngrams)


def frequent_words(list_words, ngrams_number=1, number_top_words=10):
    """
    Create n-grams for list of tokens
    Parameters
    ----------
    ngrams_number : int
    number_top_words : int
        output dataframe length
    Returns
    -------
    DataFrame
        Dataframe with the entities and their frequencies.
    """
    frequent = []
    if ngrams_number == 1:
        pass
    elif ngrams_number >= 2:
        list_words = create_ngrams(list_words, ngrams_number)
    else:
        raise ValueError("number of n-grams should be >= 1")
    counter = Counter(list_words)
    frequent = counter.most_common(number_top_words)
    return frequent


def make_word_cloud(text_or_counter, stop_words=None, background_color="black", colormap="tab20", circle=True, dw=600,
                    filename=None):
    mask = None
    if circle:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)
    if isinstance(text_or_counter, str):
        word_cloud = wordcloud.WordCloud(stopwords=stop_words,
                                         background_color=background_color,
                                         colormap=colormap,
                                         width=dw,
                                         mask=mask,
                                         mode="RGBA").generate(text_or_counter)
    else:
        if stop_words is not None:
            text_or_counter = Counter(word for word in text_or_counter if word not in stop_words)
        word_cloud = wordcloud.WordCloud(stopwords=stop_words,
                                         background_color=background_color,
                                         colormap=colormap,
                                         width=dw,
                                         mask=mask,
                                         mode="RGBA").generate_from_frequencies(text_or_counter)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    if not filename is None:
        # plt.savefig(filename)
        word_cloud.to_file(filename)
    else:
        plt.show;


def extract_words(ngrams):
    words = []
    for ng in ngrams:
        for w in ng.split():
            words.append(w)
    return words


def clean_txt(txt):
  for symbol in "().,«»;!&?:-|)":
    txt = txt.lower().strip().replace(symbol, "")
  if symbol not in "абвгдеёжзийклмнопрстуфхцчщшьыъэюя":
    txt = txt.replace(symbol, "")
  return txt


def st_wordcloud(text, stop_words):
    new_stop_words = ["запроса", "требование", "требования", "железной", "дороги",
                      "требованиям", "ржд", "должно", "возможность", "составителей",
                      "составителя", "оао", "инновационного"]
    for w in new_stop_words:
        stop_words.append(w)

    flat_list = [item for sublist in text for item in sublist]
    flat_list = [clean_txt(item) for item in flat_list if item not in stop_words and item.isalpha()]
    doc_ngrams = create_ngrams(flat_list, 4)
    doc_words = extract_words(doc_ngrams)
    cloud = dict(frequent_words(doc_words, number_top_words=100))
    make_word_cloud(cloud, colormap="Greens", background_color="#3A3A3F", filename="wc.png", circle=True, dw=300)
