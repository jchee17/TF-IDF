# Jerry Chee
# Functions used for generating tfidf scores and 
# calculating cosine distances for document comparions.

import os
import re
import subprocess
import pickle
import webbrowser
import math
from textblob import TextBlob

# Given a word and path to a corpus.
# Finds the number of documents in that corpus
#   which contain the word.
def num_inDirectory(word, path_directory): 
    process = subprocess.Popen("grep {} {} -l | wc -l".format(word, 
        path_directory+'*'),shell=True, stdout=subprocess.PIPE) 
    return int(process.communicate()[0].strip('\n'))

# Finds number of a given word in a document.
def num_inFile(word, f, path_directory):
    process = subprocess.Popen("grep {} {} -c".format(word, 
        path_directory+f), shell=True, stdout=subprocess.PIPE) 
    return int(process.communicate()[0].strip('\n'))

# Go over all documents in the corpus, 
#   import text into dictionaries
def import_corpus(path_corpus):
    # output dictionary
    doclist = {}

    for subidr, dirs, files in os.walk(path_corpus):
        for file in files:
            file_path = path_corpus + file
            shakes = open(file_path, 'r')
            text = shakes.read()
            lowers = text.lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            unicode_ignore = unicode(no_punctuation, errors='ignore')
            doclist[file] = TextBlob(unicode_ignore)
    
    return doclist

# Finds common vocab between two corpuses 
# (stored in dictionaries)
def common_vocab(doclist_wiki, doclist_arxiv):
    vocab_wiki = set()
    vocab_arxiv = set()

    for i, doc in enumerate(doclist_wiki):
        word_set = set(doclist_wiki[doc].words)
        vocab_wiki = vocab_wiki.union(word_set)

    for i, doc in enumerate(doclist_arxiv):
        word_set = set(doclist_arxiv[doc].words)
        vocab_arxiv = vocab_arxiv.union(word_set)

    vocab = vocab_wiki.intersection(vocab_arxiv)
    return vocab

# Compute the counts needed to tfidf and store in dictionaires
def preprocess_tfidf(vocab, path_wiki, path_arxiv):
    wiki_sep_byword = {}
    arxiv_sep_byword = {}
    
    wiki_sep_byfile = {}
    arxiv_sep_byfile = {}

    wiki_wordsindoc = {}
    arxiv_wordsindoc = {}

    # *_sep_byword: the number of documents which contain
    #   each word in the wiki and arxiv directories
    for word in vocab:
        wiki_sep_byword[word] = num_inDirectory(word, path_wiki)
        arxiv_sep_byword[word] = num_inDirectory(word, path_arxiv)

    wiki_sep_byword['number of documents in wiki'] = len(os.listdir(path_wiki))
    arxiv_sep_byword['number of documents in arxiv'] = len(os.listdir(path_arxiv))

    # *_sep_byfile: for each word, its count in each document
    for f in os.listdir(path_wiki):
        wiki_sep_byfile[f] = {}
        for word in vocab:
            wiki_sep_byfile[f][word] = num_inFile(word, f, path_wiki)

    for f in os.listdir(path_arxiv):
        arxiv_sep_byfile[f] = {}
        for word in vocab:
            arxiv_sep_byfile[f][word] = num_inFile(word, f, path_arxiv)

    # *_wordsindoc: the number of words in each doc 
    file = None
    txt = ''
    txt_tb = None

    for f in wiki_sep_byfile:
        file = open(path_wiki+f, 'r')
        txt = file.read()
        file.close()
        txt = unicode(txt, errors='ignore')
        txt_tb = TextBlob(txt)
        wiki_wordsindoc[f] = len(txt_tb.words)

    for f in arxiv_sep_byfile:
        file = open(path_arxiv+f, 'r')
        txt = file.read()
        file.close()
        txt = unicode(txt, errors='ignore')
        txt_tb = TextBlob(txt)
        arxiv_wordsindoc[f] = len(txt_tb.words)

    return (wiki_sep_byword, wiki_sep_byfile, wiki_wordsindoc, 
            arxiv_sep_byword, arxiv_sep_byfile, arxiv_wordsindoc)

# computes tf
def tf(word, document, corpus):
    # check string input
    if (corpus == 'wiki'):
        return (wiki_sep_byfile[document][word] / wiki_wordsindoc[document])
    elif (corpus == 'arxiv'):
        return (arxiv_sep_byfile[document][word] / arxiv_wordsindoc[document])
    else:
        return None

# computes idf
def idf(word, corpus):
    # check string input
    if (corpus == 'wiki'):
        return math.log(length_wiki / (1 + wiki_sep_byword[word]))
    elif (corpus == 'arxiv'):
        return math.log(length_arxiv/ (1 + arxiv_sep_byword[word]))
    else:
        return None

# computes tf-idf
def tfidf(word, document, corpus):
    return tf(word, document, corpus) * idf(word, corpus)

# Compute tf-idf scores and store in two matrices
def compute_tfidf(vocab, wiki_sep_byfile, arxiv_sep_byfile):
    tfidf_wiki = {}
    tfidf_arxiv = {}

    for f in wiki_sep_byfile:
        scores = [tfidf(word, f, 'wiki') for word in vocab]
        tfidf_wiki[f] = scores

    for f in arxiv_sep_byfile:
        scores = [tfidf(word, f, 'arxiv') for word in vocab]
        tfidf_arxiv[f] = scores

    return(tfidf_wiki, tfidf_arxiv)

# magnitutde of a list
def magnitude(v):
    return math.sqrt(sum(i*i for i in v))

# normalizes a list (vector)
def normalize(v):
    vmag = magnitude(v)
    if vmag == 0:
        print('vmag is zero')
        return 0
    else:
        return [ i/vmag for i in v ]

# dot product of two lists (vectors)
def dot_product(u,v):
    return sum(u[i]*v[i] for i in range(len(u)))

# normalize the dictionaries holding tf-idf scores
def normalize_dicts(tfdf_wiki, tfidf_arxiv):
    norm = []
    tfidf_wiki_norm = {}
    tfidf_arxiv_norm = {}

    for f in tfidf_wiki:
        norm = normalize(tfidf_wiki[f])
        tfidf_wiki_norm[f] = norm

    for f in tfidf_arxiv:
        norm = normalize(tfidf_arxiv[f])
        tfidf_arxiv_norm[f] = norm

    return (tfidf_wiki_norm, tfidf_arxiv_norm)

# Compute cosine distance between each arxiv papers,
#   and each wiki concept
def cosine_distance_dicts(tfidf_wiki_norm, tfidf_arxiv_norm):
    doc_comparisons = {}

    for f in tfidf_arxiv_norm:
        doc_comparisons[f] = {key:dot_product(tfidf_arxiv_norm[f], 
            tfidf_wiki_norm[key]) for key in tfidf_wiki_norm}

    return (doc_comparisons)

# Pickling function. names file variable name.p
def save(data_struct, filename, directory):
    pickle.dump(data_struct, open(direcotry+filename+".p", 'w'))

