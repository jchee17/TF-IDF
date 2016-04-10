from __future__ import unicode_literals
from __future__ import division
import os 
import re
import pickle
import numpy as np
import webbrowser

cos_dist_tfidf = np.load("./data_objects/cos_dist_tfidf.npy")
cos_dist_okapi = np.load("./data_objects/cos_dist_okapi.npy")

# tfidf_wiki = np.load("./data_objects/tfidf_wiki.npy")
# tfidf_arxiv = np.load("./data_objects/tfidf_arxiv.npy")

# okapi_bm25_wiki = np.load("./data_objects/okapi_bm25_wiki.npy")
# okpapi_bm25_arxiv = np.load("./data_objects/okapi_bm25_arxiv.npy")

list_wiki = pickle.load(open("./data_objects/list_wiki.p", 'r'))
list_arxiv = pickle.load(open("./data_objects/list_arxiv.p", 'r'))
list_vocab = pickle.load(open('./data_objects/list_vocab.p', 'r'))

def lookup_top_n(cos_dist_array, list_articles, index, n):
    """ given the index of article and cos_dist_array,
        returns the index of top n concepts
    """
    article_array = cos_dist_array[index]
                            
    # give top n partition
    sorted_ind = np.argpartition(article_array, -n)[-n:]
    # sort in this top n
    sorted_ind = sorted_ind[np.argsort(article_array[sorted_ind])[::-1]]
    return sorted_ind
    
def display_article(list_articles, index):
    """ takes in article and displays arxiv.org/abs 
        page in browser
    """
    # get id string
    id_string = re.sub("\_trunc.txt", '', list_articles[index])
    # open in browser
    webbrowser.open("http://arxiv.org/abs/" + id_string)
    
# =============================================================================
while (1):
    i = raw_input("Arxiv article from 0 to {}\n".format(len(list_arxiv)))
    i = int(i)
            
    n = 10
    top_n_tfidf = lookup_top_n(cos_dist_tfidf, list_arxiv, i, n)
    top_n_okapi = lookup_top_n(cos_dist_okapi, list_arxiv, i, n)

    print(list_arxiv[i])

    print("TFIDF")
    for j in range(n):
        k = top_n_tfidf[j]
        print("{}\t{}\t{}".format(k, list_wiki[k], round(cos_dist_tfidf[i, k],3)))
    print('\n')

    print("OKAPI")
    for j in range(n):
        k = top_n_okapi[j]
        print("{}\t{}\t{}".format(k, list_wiki[k], round(cos_dist_okapi[i, k],3)))
    print('\n')
    
    display_article(list_arxiv, i)
