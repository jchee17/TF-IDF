# Jerry Chee
# functions for keyword search by tf-idf score

import os
import re
import timeit
import string
import pickle
from nltk.tokenize import RegexpTokenizer

def concept_list(concept_fname, path):
    """ reads in a txt file, one term per line
        outputs as python list in given path
        as pickled object """
    # open file pointer
    f = open("./concepts/"+concept_name, "r")
    concepts = concept_file.readlines()
    # cp of list to iterate over 
    concepts_cp = list(concepts)
    tmp_str = ""

    for word in concepts_cp:
        tmp_str = word
        tmp_str = tmp_str.strip("\n")
        tmp_str = tmp_str.lower()

        concepts.remove(word)
        concepts.append(tmp_str)

    # from observation the concept lists all had ''
    concepts.remove('')

    # pickle 
    pickle.dump(concepts, open(path+concept_name+'.p', 'w'))

def data_process(arxiv_path, concepts, 
        arxiv_wordsindoc, arxiv_sep_byfile, arxiv_sep_byword): 
    
    articles = os.listdir(arxiv_path)

    # file pointer and txt var
    fp = None
    txt = ""
    
    for fname in articles:
        # import text
        fp = open(arxiv_path+fname, 'r')
        txt = fp.read()
        txt = txt.lower()

        # create sub dictionary
        arxiv_sep_byfile[fname] = {}

        for word in 
