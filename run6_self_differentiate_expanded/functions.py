# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import os
import re
import timeit
import math
import codecs
import pickle
from numpy import linalg as LA
import numpy as np
from nltk import RegexpTokenizer

def gen_list_corpus(path_corpus):
    print("\ngen_list_corpus:{}:".format(path_corpus))
    list_corpus = []
    for fname in os.listdir(path_corpus):
        if not fname.startswith('.'):
            list_corpus.append(fname)
    return list_corpus

def gen_vocab(vocab_fname, path):
    print("\ngen_vocab:{}".format(vocab_fname))
    """ reads in a csv file, 
        outputs as python list in given path
        as pickled object. unicode.
        Also add unigrams for every line"""
    # open file pointer
    f = codecs.open(path+vocab_fname, 'r', "utf-8")
            
    # output list
    concepts = []
                        
    # read in lines
    for line in f.readlines():
        concepts = concepts + line.lower().strip("\n").split(',')

    # from observation the concept lists all had ''
    while ('' in concepts):
        concepts.remove('')

    # add unigrams to concepts. does not preserve order of list
    unigrams = set()
    set_concepts = set(concepts)
    tokenizer = RegexpTokenizer(ur'\w+')
                
    for phrase in concepts:
        unigrams.update(tokenizer.tokenize(phrase))
                                    
    set_concepts.update(unigrams)
    return list(set_concepts)

def gen_regex_c(list_vocab):
    """ reads in list (unicode), outputs dictionary
        where key value is keyword in concept,
        value is compiled regex"""
    # create output dict
    re_vocab_c = {}

    # create regex expression for every keyword in concepts
    re_keyword = ur''
    re_match = ur"(\\-)|(\\–)|(\\\s)"
    re_replace = ur"[-\s–]"
    for keyword in list_vocab:
        re_keyword = ur'\s' + re.sub(re_match, re_replace, re.escape(keyword)) + ur'\s'
        re_vocab_c[keyword] = re.compile(re_keyword)

    return re_vocab_c

def gen_doc_term_counts(path_corpus, list_corpus, list_vocab):
    print("\ngen_doc_term_counts:{}".format(path_corpus))
    """ generates document-term matrix given a path to a 
        corpus and common vocab
    """
    num_docs = len(list_corpus)
    num_terms = len(list_vocab)
    doc_term = np.zeros((num_docs, num_terms))
    counts_corpus = np.zeros(num_docs)

    # generate (dict) compiled regex's
    re_c_vocab = gen_regex_c(list_vocab)
    tokenizer = RegexpTokenizer(ur'\w+')

    # iterate over files
    fp = None
    txt = u''
    r = None
    num = 0.0
    tokens = []
        
    count = 0
    every = 50
    start= timeit.default_timer()
    checkpoint = 0.0
    for i in range(num_docs):
        fp = codecs.open(path_corpus+list_corpus[i], 'r', "utf-8", errors="ignore")
        txt = fp.read()
        txt = txt.lower()
        fp.close()
        
        # tokenize
        tokens = tokenizer.tokenize(txt) 
        counts_corpus[i] = len(tokens)
        
        # count number terms
        for j in range(num_terms):
            r = re_c_vocab[ list_vocab[j] ]
            num = len(r.findall(txt, re.UNICODE))  
            doc_term[i,j] = num
            
        if (count % every == 0):
            checkpoint = timeit.default_timer()
            print(count, round(checkpoint-start, 2))
        count += 1
            
    
    return (doc_term, counts_corpus)
 
def gen_counts(path_corpus, list_corpus):
    """ creates np array, for each corpus file how many words
    in that document """
    # create output
    counts_corpus = np.zeros(len(list_corpus))

    fp = None
    txt = u''
    tokens = []
    tokenizer = RegexpTokenizer(ur'\w+')

    count = 0
    every = 500
    for f in list_corpus:
        # read in text
        fp = codecs.open(path_corpus+f, 'r', "utf-8", errors="ignore")
        txt = fp.read()
        txt = txt.lower()
        fp.close()

        # tokenize
        tokens = tokenizer.tokenize(txt) 
        counts_corpus[list_corpus.index(f)] = len(tokens)
        
        # count interations
        if count % every == 0:
            print(count)
        count += 1

    return counts_corpus

def gen_inv_counts(doc_term_corpus, list_corpus, list_vocab):
    print("\ngen_inv_counts")
    """ creates np array, for each vocab term how many documents 
    it can be found in.
    """
    # generate output array
    num_docs = len(list_corpus)
    num_terms = len(list_vocab)
    inv_counts = np.zeros(num_terms)
    
    count = 0
    every = 100
    for i in range(num_docs):
        for j in range(num_terms):
            if doc_term_corpus[i,j] > 0:
                inv_counts[j] += 1
        
        if count % every == 0:
            print(count)
        count += 1
            
    return inv_counts
            
def gen_tfidf(doc_term_corpus, counts_corpus, inv_counts_corpus, 
              list_corpus, list_vocab):
    print("\ngen-tfidf")
    """ generates doc-term tf-idf matrix """
    len_corpus = len(list_corpus)
    
    # output matrix
    num_docs = len(list_corpus)
    num_terms = len(list_vocab)
    tfidf_corpus = np.zeros((num_docs, num_terms))
    
    count = 0
    every = 500
    
    tf = 0.0
    idf = 0.0
    for i in range(num_docs):
        for j in range(num_terms):
            tf = doc_term_corpus[i,j] / (1 + counts_corpus[i])
            idf = math.log(len_corpus / (1 + inv_counts_corpus[j]))
            tfidf_corpus[i,j] = tf*idf
            
        if count % every == 0:
            print(count)
        count += 1
            
    return tfidf_corpus

def compute_cosine_sim(tfidf_articles, tfidf_concepts, stopwords_index):
    print("\ncomput_cosine_sim")
    """ compute doc-concept matrix of cos-dist, except
        for stopwords """
    num_docs = tfidf_articles.shape[0]
    num_concepts = tfidf_concepts.shape[0]
    cos_dist = np.zeros((num_docs, num_concepts))
    
    count = 0
    every = 100
    start = timeit.default_timer()
    checkpoint = 0.0
    
    # zero out columns of stopwords_index
    for idx in stopwords_index:
        tfidf_articles[:, idx] = 0.0
        tfidf_concepts[:, idx] = 0.0
    
    for i in range(num_docs):
        for j in range(num_concepts):
            norm = LA.norm(tfidf_articles[i]) * LA.norm(tfidf_concepts[j])
            cos_dist[i,j] = np.dot(tfidf_articles[i], tfidf_concepts[j]) / norm
            
        if (count % every == 0):
            checkpoint = timeit.default_timer()
            print(count, checkpoint-start)
        count += 1
        
    # convert nan from zero vector to zero
    cos_dist = np.nan_to_num(cos_dist)
        
    return cos_dist

def compute_conditional_cos(cos_dist, doc_term_articles, title_concepts):
    print("\ncompute_conditional_cos")
    """ sets cos_dist score to zero if title of a concept 
        not present in the article """
    num_docs = cos_dist.shape[0]
    num_concepts = cos_dist.shape[1]
    cond_cos_dist = np.copy(cos_dist)
 
    count = 0
    every = 500
    set_zero = 0
    titleIdx = 0
    for j in range(num_concepts):
        titleIdx = title_concepts[j]
        
        for i in range(num_docs):
            if doc_term_articles[i, titleIdx] == 0:
                # set cond count to 0
                cond_cos_dist[i,j] = 0

                set_zero += 1
        
        if count % every == 0:
            print(count)
        count += 1
    print(set_zero)
    return cond_cos_dist
