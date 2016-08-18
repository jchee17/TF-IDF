#!/usr/bin/env python       
# -*- coding: utf-8 -*-

import os
import codecs
import re
import pickle
import numpy as np
from analysis_functions import *

count_labels = np.load("./data_objects/count_labels.npy")
list_labels  = pickle.load(open("./data_objects/list_labels.p", 'r'))

count_wiki   = np.load("./data_objects/vocab_uni/counts_wiki_vocab_uni.npy")
list_wiki = pickle.load(open("./data_objects/list_wiki.p", 'r'))

wiki_to_concept = pickle.load(open("./data_objects/list_wiki_to_concept_names.p", 'r'))
# =============================================================================
label_ignore = gen_ignore_words(count_labels, list_labels, 90)
pickle.dump(label_ignore, open("./data_objects/label_ignore.p", 'w'))

wiki_ignore  = gen_ignore_words(count_wiki, list_wiki, 10)
pickle.dump(wiki_ignore,  open("./data_objects/wiki_ignore.p", 'w'))

(label_ignore_joined, wiki_ignore_joined) = join_ignore_words(label_ignore, 
        wiki_ignore, wiki_to_concept)
pickle.dump(label_ignore_joined, open("./data_objects/label_ignore_joined.p", 'w'))
pickle.dump(wiki_ignore_joined, open("./data_objects/wiki_ignore_joined.p", 'w'))

