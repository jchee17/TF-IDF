# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import os
import gc
import pickle
import numpy as np
from functions import *

path_wiki = "/home/jchee/hopper/corpus/bigger_ptm_data/wiki_concepts2/"
path_arxiv = "/home/jchee/hopper/corpus/bigger_ptm_data/arxiv_processed/"
list_wiki = pickle.load(open("./data_objects/list_wiki.p", 'r'))
list_arxiv = pickle.load(open("./data_objects/list_arxiv.p", 'r'))

vocab = pickle.load(open("./data_objects/vocab.p", 'r'))

counts_wiki = np.load('./data_objects/counts_wiki.npy')
counts_arxiv = np.load('./data_objects/counts_arxiv.npy')

#inv_counts_wiki = np.load('./data_objects/inv_counts_wiki.npy')
#inv_counts_arxiv = np.load('./data_objects/inv_counts_arxiv.npy')

doc_term_wiki = np.load("./data_objects/doc_term_wiki.npy")
doc_term_arxiv = np.load("./data_objects/doc_term_arxiv.npy")

#tfidf_wiki = np.load("./data_objects/tfidf_wiki.npy")
#tfidf_arxiv = np.load("./data_objects/tfidf_arxiv.npy")

#cos_dist_tfidf = np.load("./data_objects/cos_dist_tfidf.npy")

#wikiIdx_toVocabIdx = pickle.load(open("./data_objects/wikiIdx_toVocabIdx.p", 'r'))

# =============================================================================
#list_wiki = gen_list_corpus(path_wiki)
#print("len list_wiki:{}".format(len(list_wiki)))
#pickle.dump(list_wiki, open("./data_objects/list_wiki.p", 'w'))
#list_arxiv = gen_list_corpus(path_arxiv)
#print("len list_arxiv:{}".format(len(list_arxiv)))
#pickle.dump(list_arxiv, open("./data_objects/list_arxiv.p", 'w'))

#vocab = gen_vocab("phrases-bold2", "./data_objects/")
#pickle.dump(vocab, open("./data_objects/vocab.p", 'w'))

#(doc_term_wiki, counts_wiki)  = gen_doc_term_counts(path_wiki, list_wiki, vocab)
#np.save("./data_objects/doc_term_wiki.npy", doc_term_wiki)
#np.save("./data_objects/counts_wiki.npy", counts_wiki)

#(doc_term_arxiv, counts_arxiv) = gen_doc_term_counts(path_arxiv, list_arxiv, vocab)
#np.save("./data_objects/doc_term_arxiv.npy", doc_term_arxiv)
#np.save("./data_objects/counts_arxiv.npy", counts_arxiv)

inv_counts_wiki = gen_inv_counts(doc_term_wiki, list_wiki, vocab)
np.save("./data_objects/inv_counts_wiki.npy", inv_counts_wiki)

inv_counts_arxiv = gen_inv_counts(doc_term_arxiv, list_arxiv, vocab)
np.save("./data_objects/inv_counts_arxiv.npy", inv_counts_arxiv)

tfidf_wiki = gen_tfidf(doc_term_wiki, counts_wiki, inv_counts_wiki, list_wiki, vocab)
np.save("./data_objects/tfidf_wiki.npy", tfidf_wiki)

tfidf_arxiv = gen_tfidf(doc_term_arxiv, counts_arxiv, inv_counts_arxiv, list_arxiv, vocab)
np.save("./data_objects/tfidf_arxiv.npy", tfidf_arxiv)

# give top n partition
n = int(0.01 * len(inv_counts_arxiv))
#stopwords_index = np.argpartition(inv_counts_arxiv, -n)[-n:]
# sort in this top n
#stopwords_index = stopwords_index[np.argsort(inv_counts_arxiv[stopwords_index])[::-1]]
#cos_dist = compute_cosine_sim(tfidf_arxiv, tfidf_wiki, stopwords_index)
#np.save("./data_objects/cos_dist_tfidf.npy", cos_dist)

#self_diff_stopwords = compute_cosine_sim(tfidf_wiki, tfidf_wiki, stopwords_index)
#np.save("./data_objects/self_diff_stopwords.npy", self_diff_stopwords)

#self_diff = compute_cosine_sim(tfidf_wiki, tfidf_wiki, [])
#np.save("./data_objects/self_diff.npy", self_diff)

cond_cos_dist = compute_conditional_cos(cos_dist, doc_term_arxiv, wikiIdx_toVocabIdx)
np.save("./data_objects/cond_cos_dist.npy", cond_cos_dist)
