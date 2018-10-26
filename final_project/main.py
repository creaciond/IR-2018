from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
import json
import pandas as pd

import warnings
warnings.simplefilter("ignore")

from flask import Flask
from flask import url_for, render_template, request
from search_methods import search_inv_index, search_w2v, search_d2v

app = Flask(__name__)

path_to_corpus = "./Avito_Beauty_Corpus"


def search(query_raw, search_method):
    if search_method == "inv_index":
        search_result = search_inv_index(query_raw)
    elif search_method == "w2v":
        search_result = search_w2v(query_raw)
    elif search_method == "d2v":
        search_result = search_d2v(query_raw)
    else:
        raise TypeError("Unsupported search method, please try again. " + \
            "Supported methods are: " + \
            "\n\t\t- inverted index,\n\t\t- word2vec,\n\t\t- doc2vec")
    return search_result


def prettify_result(search_result):
    prettified_result = []
    for item in search_result:
        link = "https://www.avito.ru/moskva/krasota_i_zdorove/" + item
        text_path = path_to_corpus + "/" + item + ".txt"
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        prettified_result.append((link, text))
    return prettified_result


@app.route("/")
def index():
    if request.args:
        # 1. query
        query = request.args["query"]
        # 2. search types
        type = request.args["search_mode"]
        # perform search depending on type
        print("tried to search {}, {} type".format(query, type))
        results = search(query, type)
        print(results)
        results_to_render = prettify_result(results)
        # render search results
        return render_template("results.html", result_list=results_to_render)
    # render initial page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)