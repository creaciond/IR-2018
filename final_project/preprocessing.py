import re
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from pymorphy2 import MorphAnalyzer

reg_punct = re.compile("[«–»—!\$%&'()*+,./:;<=>?@^_`{|}~']*")
reg_num = re.compile("[0-9]+")
reg_latin = re.compile("[a-z]+")

stops = stopwords.words("russian")
trash = set([" ", "-", "\n", ""])
morph = MorphAnalyzer()


def preprocessing(text, *option):
    """Given a text, makes it appropriate for later work:
        1. converts to lowercase
        2. removes whitespace chars
        3. removes punctuation marks
        4. removes latin symbols

    :param str text: original text to preprocess
    :param str option: if "w2v", removes stopwords,
        otherwise leaves everything as is

    :returns: str proc_text: processed text
    """
    proc_text = text.lower()

    proc_text = proc_text.replace("\n", " ")
    proc_text = reg_punct.sub("", proc_text)
    proc_text = reg_num.sub("", proc_text)
    proc_text = reg_latin.sub("", proc_text)

    tokens = wordpunct_tokenize(proc_text)
    raw_lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    if option == "w2v":
        lemmas = [lemma for lemma in raw_lemmas if (lemma not in stops) and (lemma not in trash)]
    else:
        lemmas = [lemma for lemma in raw_lemmas if lemma not in trash]
    return " ".join(raw_lemmas)