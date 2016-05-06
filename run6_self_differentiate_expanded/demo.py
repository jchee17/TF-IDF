# Jerry Chee
# run a demo to lookup top concepts given tf-idf python dictionaries
# -*- encoding: utf-8 -*-

from __future__ import division, unicode_literals

import os
import re
import webbrowser
import pickle
import numpy as np
from numpy import linalg as LA
import operator # change these paths 

list_wiki  = pickle.load(open("./data_objects/list_wiki.p", 'r'))
list_arxiv = pickle.load(open("./data_objects/list_arxiv.p", 'r'))
list_vocab = pickle.load(open("./data_objects/vocab.p", 'r'))

#list_wiki_trunc = pickle.load(open("../run5_conditional_boldphrases/data_objects/list_wiki.p", 'r'))
#list_arxiv_trunc = pickle.load(open("../run5_conditional_boldphrases/data_objects/list_arxiv.p", 'r'))
#list_vocab_trunc = pickle.load(open("../run5_conditional_boldphrases/data_objects/vocab.p", 'r'))



tfidf_wiki = np.load("./data_objects/tfidf_wiki.npy")
tfidf_arxiv = np.load("./data_objects/tfidf_arxiv.npy")
#tfidf_wiki_trunc = np.load("../run5_conditional_boldphrases/data_objects/tfidf_wiki.npy")
#tfidf_arxiv_trunc = np.load("../run5_conditional_boldphrases/data_objects/tfidf_arxiv.npy")

cos_dist   = np.load("./data_objects/cos_dist_tfidf.npy")
#cos_dist_trunc = np.load("../run5_conditional_boldphrases/data_objects/cos_dist_tfidf.npy")
#cond_cos_dist = np.load("./data_objects/cond_cos_dist.npy")

# =============================================================================
# function to take in a arxiv paper name and output the top k wiki concepts
def relevant_concepts(doc_comparison, tfidf_articles, tfidf_concepts, list_articles, list_concepts, list_common_vocab, article_idx, k):
    print("Top concepts in document {}:".format(article_idx))
 
    if article_idx in range(len(list_articles)):
        # check if article_idx in doc_comparsions
        scores          = doc_comparison[article_idx]
        sorted_idx      = np.argpartition(scores, -k)[-k:]
        sorted_idx      = sorted_idx[np.argsort(scores[sorted_idx])[::-1]]
        for idx in sorted_idx:
            print("\tConcept: {}, score: {}".format(list_wiki[idx], 
                round(scores[idx], 5)))
            article_vec = tfidf_articles[article_idx]
            concept_vec = tfidf_concepts[idx]
            norm_prod   = np.divide(np.multiply(article_vec, concept_vec), LA.norm(article_vec)*LA.norm(concept_vec))
            sorted_idx2 = np.argpartition(norm_prod, -5)[-5:]
            sorted_idx2 = sorted_idx2[np.argsort(norm_prod[sorted_idx2])[::-1]]
            for idx2 in sorted_idx2:
                print("\t\t word = {}, score = {}".format(list_common_vocab[idx2], round(norm_prod[idx2],2)))
    else:
        print("\tError: input doc not in given dictionary")

def relevant_concepts2(doc_comparison, list_articles, list_concepts, list_common_vocab, article_idx, k):
    print("Top concepts in document {}:".format(article_idx))
 
    if article_idx in range(len(list_articles)):
        # check if article_idx in doc_comparsions
        scores          = doc_comparison[article_idx]
        sorted_idx      = np.argpartition(scores, -k)[-k:]
        sorted_idx      = sorted_idx[np.argsort(scores[sorted_idx])[::-1]]
        for idx in sorted_idx:
            print("\tConcept: {}, score: {}".format(list_wiki[idx],#.decode('utf-8'), 
                round(scores[idx], 5)))
    else:
        print("\tError: input doc not in given dictionary")

def show_doc_text(article_idx):
    f = open(path_arxiv + article_idx, 'r')
    text = f.read()
    f.close()
    print('\n' + 'Title: ' + article_idx + '\n\n' + text)

def show_doc_inbrowser(article_str):
    # first get the article id from string input
    #id_string = re.sub('\_trunc.txt', '', article_str)
    id_string = re.sub('\.txt', '', article_str)
    webbrowser.open('http://arxiv.org/abs/' + id_string)

# =============================================================================
# main loop
k = 15
while(1):
    i = raw_input("Enter an integer from 0 to {} to access one of the articles.\n".format(len(list_arxiv)))
    i = int(i)
    
    #ar_str_trunc = re.sub('\.txt', '_trunc.txt', list_arxiv[i])
    #if ar_str_trunc in list_arxiv_trunc:
    #i_trunc = list_arxiv_trunc.index(ar_str_trunc)

    print("phrases: cosine distance tf-idf score")
    relevant_concepts(cos_dist, tfidf_arxiv, tfidf_wiki, list_arxiv, list_wiki, list_vocab, i, k)
    #print("phrases: cosinse distance tf-idf score [trunc]")
    #relevant_concepts(cos_dist_trunc, tfidf_arxiv_trunc, tfidf_wiki_trunc, list_arxiv_trunc, list_wiki_trunc, list_vocab_trunc, i_trunc, k)
    
    #print("phrases: cosine distance tf-idf score conditional")
    #relevant_concepts(cond_cos_dist, tfidf_arxiv, tfidf_wiki, list_arxiv, list_wiki, list_vocab, i, k)
    show_doc_inbrowser(list_arxiv[i])
    print('\n');

    #else:
        #print("another try")
