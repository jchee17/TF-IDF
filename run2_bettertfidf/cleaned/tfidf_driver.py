# Jerry Chee
# 1/17/16
# Driver to run tfidf_functions.
# Takes in paths to corpus of wiki pages and arxiv papers.

import os
import pickle
import tfidf_functions.py

# paths for input. change accordingly
path_wiki = '/home/jerry/Data/Hopper_Project/ptm_data/wiki/'
path_arxiv = '/home/jerry/Data/Hopper_Project/ptm_data/arxiv_processed_trunc/'

# we define the data structures needed
# dictionaries hold text for wiki pages and arxiv papers
doclist_wiki = {}
doclist_arxiv = {}

# common vocabulary
vocab = set()

# data structures to pre-compute tf-idf scores
wiki_sep_byword = {}
arxiv_sep_byword = {}

wiki_sep_byfile = {}
arxiv_sep_byfile = {}

wiki_wordsindoc = {}
arxiv_wordsindoc = {}

# dictionaries store tf-idf score vectors for all wiki and arxiv
tfidf_wiki = {}
tfidf_arxiv = {}

# normalized vectors of tf-idf scores
tfidf_wiki_norm = {}
tfidf_arxiv_norm = {}

# dictionary holding cosine distances between tf-idf score vectors
doc_comparisons = {}

# =============================================================================
doclist_wiki = import_corpus(path_wiki)
doclist_arxiv = import_corpus(path_arxiv)

vocab = common_vocab(doclist_wiki, doclist_arxiv)

preprocess = preprocess_tfidf(vocab, path_wiki, path_arxiv)

wiki_sep_byword = preprocess[0]
wiki_sep_byfile = preprocess[1]
wiki_wordsindoc = preprocess[2]

arxiv_sep_byword = preprocess[3]
arxiv_sep_byfile = preprocess[4]
arxiv_wordsindoc = preprocess[5]

tfidf = compute_tfidf(vocab, wiki_sep_byfile, arxiv_sep_byfile)
tfidf_wiki = tfidf[0]
tfidf_arxiv = tfidf[1]

tfidf_norm = normalize_dicts(tfidf_wiki, tfidf_arxiv)
tfidf_wiki_norm = tfidf_norm[0]
tfidf_arxiv_norm = tfidf_norm[1]

doc_comparisons = cosine_distance_dicts(tfidf_wiki_norm, tfidf_arxiv_norm)

# =============================================================================
# saving
save(vocab, "vocab.p", "./")

save(wiki_sep_byword, "wiki_sep_byword.p", "./")
save(wiki_sep_byfile, "wiki_sep_byfile.p", "./")
save(wiki_wordsindoc, "wiki_wordsindoc.p", "./")

save(arxiv_sep_byword, "arxiv_sep_byword.p", "./")
save(arxiv_sep_byfile, "arxiv_sep_byfile.p", "./")
save(arxiv_wordsindoc, "arxiv_wordsindoc.p", "./")

save(tfidf_wiki, "tfidf_wiki.p", "./")
save(tfidf_arxiv" tfidf_arxiv.p", "./")

save(doc_comparisons, "doc_comparisons.p", "./")
