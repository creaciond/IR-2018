from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim import matutils
import json
from math import log
import numpy as np
import os
import pandas as pd
from statistics import mean

from preprocessing import preprocessing

import warnings
warnings.simplefilter("ignore")

# global term_doc_matrix
# global inv_index
# global d2v_index
# global d2v_model
# global w2v_index
# global w2v_model

# word2vec
w2v_model = Word2Vec.load("./data/w2v/araneum_none_fasttextcbow_300_5_2018.model")
print("Word2Vec model loaded successfully")
with open("./data/w2v_data.json", "r", encoding="utf-8") as w2v_base:
    w2v_index = json.load(w2v_base)
print("Word2Vec index loaded successfully")

# doc2vec
d2v_model = Doc2Vec.load("./data/d2v_avito.model")
print("Doc2Vec model loaded successfully")
with open("./data/d2v_data.json", "r", encoding="utf-8") as d2v_base:
    d2v_index = json.load(d2v_base)
print("Doc2Vec index loaded successfully")

# inv index
term_doc_matrix = pd.read_csv("./data/term_doc_matrix.csv", sep=";", encoding="utf-8")
print("Term-doc matrix loaded successfully")
with open("./data/inv_index.json", "r", encoding="utf-8") as src_inv_index:
    inv_index = json.load(src_inv_index)
print("Inverted index loaded successfully")


"""===============================================================================
                        Inverted Index
==============================================================================="""


def count_avgdl(term_doc_matrix):
    doc_length_total = []
    for item in term_doc_matrix.index.values:
        doc_length = int(term_doc_matrix[term_doc_matrix.index == item].sum(axis=1))
        doc_length_total.append(doc_length)
    avgdl = mean(doc_length_total)
    return doc_length_total, avgdl


def search_single_term(query):
    try:
        res = term_doc_matrix.index[term_doc_matrix[query] != 0].tolist()
        return res
    except:
        return []


def compute_idf(query, N):
    n_q = len(inv_index[query])
    frac = (N - n_q + 0.5)/(n_q + 0.5)
    idf = log(frac)
    return idf


def okapi_bm25(idf, freq, D, avgdl):
    k1 = 2.0
    b = 0.75
    score = idf * (k1 + 1) * freq / (freq + k1 * (1 - b + b * D / avgdl))
    return score


# static vars for inv_index
doc_length_total, avgdl = count_avgdl(term_doc_matrix)
text_ids = [item[:-4] for item in os.listdir("./Avito_Beauty_Corpus") if item != ".DS_Store"]
vocab = term_doc_matrix.columns.values
N = len(term_doc_matrix.index)


def search_inv_index(raw_query):
    query = preprocessing(raw_query)
    results = {ad: 0 for ad in text_ids}

    for word in query.split():
        if word in set(vocab):
            for i, ad in enumerate(text_ids):
                if ad in search_single_term(word):
                    freq = inv_index[word][ad]
                else:
                    freq = 0
                D = doc_length_total[i]
                idf = compute_idf(word, N)
                bm25 = okapi_bm25(float(idf), float(freq), int(D), avgdl)
                results[ad] += bm25
    sort_res = sorted(results.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [res[0] for res in sort_res]


"""===============================================================================
                        Doc2Vec
==============================================================================="""


def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1).astype(float))
    v2_norm = matutils.unitvec(np.array(v2).astype(float))
    return np.dot(v1_norm, v2_norm)


def get_d2v_vectors(document, d2v_index, file_index):
    """Transforms a document into the Doc2Vec vector and saves
    this information in the index.

    :param str document: document to be transformed
    :param dict d2v_index: dict used to index the corpus for
    further search

    :returns: dict d2v_index: updated index
        file_index (str) - vector (list of float)
    """
    vec = d2v_model.infer_vector(document)
    d2v_index[file_index] = list(vec)
    return d2v_index


def search_d2v(raw_query):
    query = preprocessing(raw_query)
    query_vec = d2v_model.infer_vector(query)
    results = {}
    for item in d2v_index:
        this_doc_vec = d2v_index[item]
        results[item] = similarity(query_vec, this_doc_vec)
    sort_res = sorted(results.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [res[0] for res in sort_res]


"""===============================================================================
                        Word2Vec
==============================================================================="""


def get_w2v_vectors(text, model):
    """Получает вектор документа"""
    total_counter = 0
    total_vector = np.zeros(300)
    for word in str(text).split():
        try:
            vector = np.array(model.wv[word])
            total_vector += vector
            total_counter += 1
        except:
            continue
    res_vector = total_vector / total_counter
    return res_vector


def search_w2v(query):
    query_vec = get_w2v_vectors(w2v_model, str(query))
    results = {}
    for item in w2v_index:
        this_doc_vec = w2v_index[item]
        results[item] = similarity(query_vec, this_doc_vec)
    sort_res = sorted(results.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return [res[0] for res in sort_res]
