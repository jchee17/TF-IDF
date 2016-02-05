# Jerry Chee

from __future__ import division, unicode_literals

import os
import re
import webbrowser
import pickle

path_arxiv = '/home/jerry/Data/Hopper_Project/ptm_data/arxiv_processed_trunc/'
path_wiki = '/home/jerry/Data/Hopper_Project/ptm_data/wiki/'

# load cos dist comparisons (prev run)
doc_comparisons = pickle.load(open('../run2_bettertfidf/rough/data_objects/doc_comparisons.p', 'r'))
# load tf-idf keyword scores (this run)
tfidf_arxiv = pickle.load(open('./data_objects/tfidf_arxiv_caseinsen.p', 'r'))

# load sep concept lists
numTh = pickle.load(open('./data_objects/numberTheory-concepts.p', 'r'))
opt = pickle.load(open('./data_objects/opt-concepts.p', 'r'))
prob = pickle.load(open('./data_objects/prob-concepts.p', 'r'))
stat = pickle.load(open('./data_objects/stat-concepts.p', 'r'))

# =============================================================================

# function to take in a arxiv paper name and output the top k wiki concepts
def relevant_concepts(arxiv_doc, dictionary, k):
    print("Top concepts in document {}:".format(arxiv_doc))
    
    # check if arxiv_doc in doc_comparsions
    if arxiv_doc in dictionary.keys(): 
        scores = dictionary[arxiv_doc]
        sorted_words = sorted(scores.items(), 
                key=lambda x : x[1], reverse = True)
        for concept, score in sorted_words[:k]:
            print("\tConcept: {} ({})".format(concept, 
                concept_member(concept)))
            #print("\tConcept: {}, score: {}".format(concept, 
                #round(score, 5)))
    
    else:
        print("\tError: input doc not in given dictionary")
        
def show_doc_text(arxiv_doc):
    f = open('/home/jerry/Data/Hopper_Project/ptm_data/arxiv_processed_trunc/' + arxiv_doc, 'r')
    text = f.read()
    f.close()
    print('\n' + 'Title: ' + arxiv_doc + '\n\n' + text)

def show_doc_inbrowser(arxiv_doc):
    # first get the article id from string input
    id_string = re.sub('\_trunc.txt', '', arxiv_doc)
    webbrowser.open('http://arxiv.org/abs/' + id_string)
    #webbrowser.open('http://arxiv.org/pdf/' + id_string + '.pdf')

def concept_member(word):
    """takes in a given word and returns out which concept
    list a member of"""
    ret_str = ""

    if (word in numTh):
        ret_str = "numTh"
    elif (word in opt):
        ret_str = "opt"
    elif (word in prob):
        ret_str = "prob"
    elif (word in stat):
        ret_str = "stat"
    else:
        ret_str = ""#"not a member of numTh, opt, prob, or stat"

    return ret_str


# =============================================================================

articles = tfidf_arxiv.keys()
indices = list(xrange(len(articles)))

articles_old = doc_comparisons.keys()
indices_old = list(xrange(len(articles)))

# main loop
while(1):
    i = raw_input("There are {} analyzed arxiv articles under both methods of cosine distance and keyword tf-idf score.\nEnter an integer from 0 to {} to access one of the articles.\n".format(len(articles_old),len(articles_old)-1))
    i = int(i)
    if i not in indices_old:
        print("Input not in range. Please try again.\n")
    else:
        print("keyword tf-idf score")
        relevant_concepts(articles_old[i], tfidf_arxiv, 5)
        print("")
        print("cosine distance tf-idf score")
        relevant_concepts(articles_old[i], doc_comparisons, 5)
        show_doc_inbrowser(articles_old[i])
        print('\n');

