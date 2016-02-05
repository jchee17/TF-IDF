# Jerry Chee

import os
import subprocess
import pickle
import math
import nltk.tokenize

# won't be using grep to preprocess tf-idf scores
# more efficient if just go through each document once, 
# keep running word counts and doc lengths.

def upload_concepts():
    """ uploades ./concepts into a list"""

    concept_file = open("./concepts", "r")
    concepts = concept_file.readlines()
    concepts_cp = list(concepts)
    tmp_str = ""

    for word in concepts_cp:
        tmp_str = word
        tmp_str = tmp_str.strip("\n")
        tmp_str = tmp_str.lower()

        concepts.remove(word)
        concepts.append(tmp_str)
    
    return concepts
