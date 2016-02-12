# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Jerry Chee
# functions for keyword search by tf-idf score
import os
import re
import timeit
import math
import codecs
import string
import pickle
from nltk.tokenize import RegexpTokenizer

def check_exist(fname, path):
    """ checks if fname exists in path"""
    return fname in os.listdir(path)

def concept_list(concept_fname, path_concepts, path_save):
    """ reads in a txt file, one term per line.
        outputs as python list in given path
        as pickled object. unicode """
    # open file pointer
    f = codecs.open(path_concepts+concept_fname, 'r', "utf-8")
    
    # output list
    concepts = []

    # read in lines
    for line in f.readlines():
        #print(line)
        concepts.append(line.lower().strip("\n"))

    # from observation the concept lists all had ''
    while ('' in concepts):
        concepts.remove('')

    # pickle 
    pickle.dump(concepts, open(path_save+concept_fname+'.p', 'w'))


def gen_regex(concepts):
    """ reads in list (unicode), outputs dictionary
        where key value is keyword in concept,
        value is regex"""
    # create output dict
    re_concepts = {}

    # create regex expression for every keyword in concepts
    re_keyword = ur''
    re_match = ur"(\\-)|(\\–)|(\\\s)"
    re_replace = u"[-\s–]"
    for keyword in concepts:
        re_keyword = re.sub(re_match, re_replace, re.escape(keyword))
        re_concepts[keyword] = re_keyword

    return re_concepts


def data_process(path_arxiv, re_concepts, path_save): 
    """ pickles the 3 dictionaries needed to compute 
        tf-idf scores. """
    # dictionaries
    arxiv_wordsindoc = {}
    arxiv_sep_byfile = {}
    arxiv_sep_byword = {}
    
    # misc and timing var
    articles = os.listdir(path_arxiv)
    num_r = 0
    start = timeit.default_timer()
    elapsed = 0
    count = 0

    # reused runtime var
    fp = None
    txt = u''
    tokens = []
    tokenizer = RegexpTokenizer(ur'\w+')

    for f in articles:
        # import text
        fp = codecs.open(path_arxiv+f, 'r', "utf-8", errors="ignore")
        txt = fp.read()

        # tokenize the article
        tokens = tokenizer.tokenize(txt)

        # save the word count to dict
        arxiv_wordsindoc[f] = len(tokens)

        # create sub dictionary
        arxiv_sep_byfile[f] = {}

        # find word counts in article
        for keyword, r in re_concepts.iteritems():
            #print(r)
            #print(txt)
            num_r = len(re.findall(r, txt, re.UNICODE))

            if (num_r > 0):
                arxiv_sep_byfile[f][keyword] = num_r

        if (count % 100 == 0):
            elapsed = timeit.default_timer() - start
            start = timeit.default_timer()
            print("{} articles complete, {}s elapsed".format(count, elapsed))
        count += 1

    # now let's comput byword
    for keyword in re_concepts:
        arxiv_sep_byword[keyword] = 0
    for f in arxiv_sep_byfile:
        for keyword in arxiv_sep_byfile[f]:
            arxiv_sep_byword[keyword] += 1


    # let's pickle them all
    pickle.dump(arxiv_sep_byfile, open(path_save+"arxiv_sep_byfile.p", 'w'))
    pickle.dump(arxiv_sep_byword, open(path_save+"arxiv_sep_byword.p", 'w'))
    pickle.dump(arxiv_wordsindoc, open(path_save+"arxiv_wordsindoc.p", 'w'))


# tf-idf functions. specific to arxiv only
def tf(keyword, doc, corpus_byfile, corpus_wordsindoc):
    """ computes term frequency"""
    # must first check if word in document
    if (keyword in corpus_byfile[doc]):
        return (corpus_byfile[doc][keyword] / corpus_wordsindoc[doc])
    else:
        return 0

def idf(keyword, corpus_byword, len_corpus):
    """ computes inverse doc frequency """
    return math.log(len_corpus / (1 + corpus_byword[keyword]))

def tfidf(path, keyword, doc, corpus_byfile, corpus_byword, 
        corpus_wordsindoc):
    """ calls to tf() and idf(), multiplies"""
    len_corpus = len(os.listdir(path))
    return tf(keyword, doc, corpus_byfile, 
            corpus_wordsindoc) * idf(keyword, corpus_byword, len_corpus)

def compute_tfidf(corpus_path, path_save, corpus_byfile, 
        corpus_byword, corpus_wordsindoc):
    """ computes tfidf scores for all keywords, all articles"""
    # output dict
    tfidf_arxiv = {}

    score = 0
    for f in corpus_byfile:
        # init sub dictionary
        tfidf_arxiv[f] = {}

        for keyword in corpus_byfile[f]:
            score = tfidf(corpus_path, keyword, f, corpus_byfile, 
                    corpus_byword, corpus_wordsindoc)

            # only save if score is meaningful
            if (score > 0):
                tfidf_arxiv[f][keyword] = score

    # pickle
    pickle.dump(tfidf_arxiv, open(path_save+"tfidf_arxiv.p", 'w'))

