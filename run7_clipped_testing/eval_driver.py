#-*- coding: utf-8 -*-  
from __future__ import unicode_literals
from __future__ import division
import os
import re
import timeit
import codecs
import pickle
import numpy as np
from eval_functions import *

path_uni  = "./data_objects/vocab_uni/"
path_bold = "./data_objects/vocab_bold/"
path_bu   = "./data_objects/vocab_bold_uni/"

# =============================================================================
#reorder(path_uni, "cos_dist_stopwords_vocab_uni")
#reorder(path_uni, "cos_dist_vocab_uni")
#reorder(path_bold, "cos_dist_stopwords_vocab_bold")
#reorder(path_bold, "cos_dist_vocab_bold")
#reorder(path_bu,   "cos_dist_stopwords_vocab_bold_uni")
#reorder(path_bu,   "cos_dist_vocab_bold_uni")


list_arxiv = pickle.load(open("./data_objects/list_arxiv.p", 'r'))
idx_wiki_to_concept_names = pickle.load(open("./data_objects/idx_wiki_to_concept_names.p", 'r'))
label_ignore = pickle.load(open("./data_objects/label_ignore_joined.p", 'r'))
wiki_ignore  = pickle.load(open("./data_objects/wiki_ignore_joined.p", 'r'))

labled_docs = "labled_docs"

list_N = [5,10,20,30]
list_T = [0.1, 0.2, 0.3, 0.4]
list_vocab = [ ("reorder_cos_dist_stopwords_vocab_uni", "vocab_uni/"),
        ("reorder_cos_dist_vocab_uni", "vocab_uni/"),
        ("reorder_cos_dist_stopwords_vocab_bold", "vocab_bold/"),
        ("reorder_cos_dist_vocab_bold", "vocab_bold/"),
        ("reorder_cos_dist_stopwords_vocab_bold_uni", "vocab_bold_uni/"),
        ("reorder_cos_dist_vocab_bold_uni", "vocab_bold_uni/") ]

for vocab in list_vocab:
    for N in list_N:
        top_N_2(N, labled_docs, vocab[0], vocab[1], list_arxiv, idx_wiki_to_concept_names, label_ignore, wiki_ignore)

for vocab in list_vocab:
    for T in list_T:
        threshold_T_2(T, labled_docs, vocab[0], vocab[1], list_arxiv, idx_wiki_to_concept_names, label_ignore, wiki_ignore)
