#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import re
import pickle
import timeit
import multiprocessing
from multiprocessing import Pool

# change accordingly
impure_doc_dir = "/home/jchee/hopper/corpus/bigger_ptm_data/arxiv_processed/"
clipped_doc_dir = "/home/jchee/hopper/corpus/bigger_ptm_data/arxiv_processed_clipped/"

# list of files in impure_doc_dir
impure_doc_list  = pickle.load(open("./data_objects/list_arxiv.p", 'r')) 
# list of bold phrases that you gave me
phrases_bold = codecs.open("./data_objects/phrases-bold2", 'r', "utf-8", errors="ignore")

# =============================================================================
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

def clipper(file_name):
    doc = codecs.open(os.path.join(impure_doc_dir, file_name), 'r', 
            "utf-8", errors="ignore").read().lower()
    doc_concepts = []
    # first search for all occurences of concept
    for concept in concept_names:
        try:
            r = re_concept_names[concept]
            if r.search(doc, re.UNICODE):
                doc_concepts.append(concept)
        except:
            pass
    # then delete all occurences of concept
    sorted_concepts = sorted(doc_concepts, key=len)[::-1]
    for concept in sorted_concepts:
       try:
           r = re_concept_names[concept]
           doc = r.sub('', doc, re.UNICODE)
           #doc = doc.replace(concept, '')
       except:
            pass
    # write to file
    with codecs.open(os.path.join(clipped_doc_dir, file_name), 'w', 
            "utf-8", errors="ignore") as clipped_file:
        clipped_file.write(doc)
    l.acquire()
    outfile.write(file_name + '\t' + ','.join(doc_concepts) + '\n')
    l.release()

# =============================================================================
concept_names = [ line.rstrip('\n').lower().split(',',1)[0] for line in phrases_bold if len(line.rstrip('\n')) > 0]
re_concept_names = gen_regex_c(concept_names)

print "Total Files = {}".format(len(concept_names))

l = multiprocessing.Lock()
with codecs.open("labled_docs.txt", 'w', "utf-8", errors="ignore") as outfile:
    pool = Pool()
    pool.map(clipper, impure_doc_list)
    pool.close()
    pool.join()
