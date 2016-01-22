# Jerry Chee
# 1/6/16
# Takes in directories of wiki and arxiv text, 
# saves in ./data_objects the data structures 
# needed to perform analysis

from __future__ import division, unicode_literals
import os
import random
import string
import timeit
import pickle
import nltk
import math
from textblob import TextBlob

date = "1_6"
# paths for input
path_arxiv = '/home/jerry/Data/Hopper_Project/ptm_data/arxiv_processed_trunc/'
path_wiki = '/home/jerry/Data/Hopper_Project/ptm_data/wiki/'

# define some global data structures/var
time_doclist = 0
time_vocab = 0
time_tfidf_wiki = 0
time_tfidf_arxiv = 0
time_norm = 0
time_cos = 0

doclist_wiki = {}
doclist_arxiv = {}
vocab = set()

tfidf_wiki = {}
tfidf_arxiv = {}

tfidf_wiki_norm = {}
tfidf_arxiv_norm = {}

doc_comparisons = {}

# define functions for tf-idf score
# note: dictionary implementation
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in bloblist[blob]) # bc dictionary

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

# go over all documents, import text into dictionaries
print("Looping through all documents, importing into dictionaries...\n")
start_time = timeit.default_timer()
for subidr, dirs, files in os.walk(path_wiki):
    for file in files:
        file_path = path_wiki + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        unicode_ignore = unicode(no_punctuation, errors='ignore')
        doclist_wiki[file] = TextBlob(unicode_ignore)
print("\tdoclist_wiki done\n")

for subidr, dirs, files in os.walk(path_arxiv):
    for file in files:
        file_path = path_arxiv + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        unicode_ignore = unicode(no_punctuation, errors='ignore')       
        doclist_arxiv[file] = TextBlob(unicode_ignore) 
time_doclist = timeit.default_timer() - start_time
print("\tdoclist_arxiv done, {}\n".format(time_doclist))

# get common vocab
#print("Finding the common vocab...\n")
#start_time = timeit.default_timer()
#vocab_wiki = set()
#vocab_arxiv = set()

#for i, doc in enumerate(doclist_wiki):
#    word_set = set(doclist_wiki[doc].words)
#    vocab_wiki = vocab_wiki.union(word_set)

#for i, doc in enumerate(doclist_arxiv):
#    word_set = set(doclist_arxiv[doc].words)
#    vocab_arxiv = vocab_arxiv.union(word_set)

#vocab = vocab_wiki.intersection(vocab_arxiv)
#time_vocab = timeit.default_timer() - start_time
#print("done {}\n".format(time_vocab))

# convert vocab to list, save to csv for human-readable
#file_vocab_list = open('./data_objects/vocab_{}.txt'.format(date), 'w')
#for i in vocab:
#    file_vocab_list.write('%s\n' % i)

# now we pickle the vocab set 
#pickle.dump(vocab, open('./data_objects/vocab_{}.p'.format(date), 'w') )

# load the vocab
vocab = pickle.load(open('./data_objects/vocab_1_6.p', 'r'))

# now let's populate the dictionaries with the scores
print("Populating dictionaries with scores...\n")
start_time = timeit.default_timer()
for i, doc in enumerate(doclist_wiki):
    scores = {word: tfidf(word, doclist_wiki[doc], doclist_wiki) 
            for word in vocab}
    tfidf_wiki[doc] = scores
time_tfidf_wiki = timeit.default_timer() - start_time

# pickle
pickle.dump(tfidf_wiki, 
    open('./data_objects/tfidf_wiki_{}.p'.format(date), 'w'))

print("\ttfidf_wiki done, {}\n".format(time_tfidf_wiki))

# we want to divide the doclist_arxiv into random subsets of 
# around size 100. Two reasons: counting tf-idf scores across
# doclist_arxiv takes too long because it is ~3600 items, compared
# to the ~70 items in doclist_wiki. Also, it may be that the
# overall tf-idf scores of doclist_arxiv will be lower because 
# because being unique in a set of 3600 is harder than in a smaller
# set.
start_time = timeit.default_timer()

doclist_arxiv_subsets = []
doclist_arxiv_cp = doclist_arxiv.copy()

len_arxiv = len(doclist_arxiv)
n = 100
subset_size = 100
i = len_arxiv

def choose_subset(dictionary, size):
    # choose random keys
    keys = random.sample(dictionary, size)

    # get list of assoicated values
    values = [doclist_arxiv_cp[i] for i in keys]

    # create new dictionary from these keys
    tmp_dict = dict( (keys[i], values[i]) 
            for i in range(len(keys)) )

    # add dict subset to list
    doclist_arxiv_subsets.append(tmp_dict)

    # remove this subset from master
    for i in keys:
        del doclist_arxiv_cp[i]

while i >= 0:
    if (i > n):
        subset_size = n
        choose_subset(doclist_arxiv_cp, subset_size)
    else:
        subset_size = i
        choose_subset(doclist_arxiv_cp, subset_size)

    i -= n

for i in range(len(doclist_arxiv_subsets)):
    print("doclist_arxiv_subsets[{}]".format(i))
    doclist = doclist_arxiv_subsets[i]

    for j, doc in enumerate(doclist):
        scores = {word: tfidf(word, doclist[doc], doclist)
                for word in vocab}
        tfidf_arxiv[doc] = scores
    
    # incremental pickling
    pickle.dump(tfidf_arxiv, 
        open('./data_objects/tfidf_arxiv_{}.p'.format(date), 'w'))

time_tfidf_arxiv = timeit.default_timer() - start_time
print("\ttfidf_arxiv done, {}\n".format(time_tfidf_arxiv))


# now compute cosine distances
# normalization function
def magnitude(v):
    return math.sqrt(sum(i*i for i in v))

def normalize(v):
    vmag = magnitude(v)
    if vmag == 0:
        print('vmag is zero')
        return 0
    else:
        return [ i/vmag for i in v ]

def dot_product(u,v):
    return sum(u[i]*v[i] for i in range(len(u)))

# normalize the two dictionaries
start_time = timeit.default_timer()
for key in tfidf_wiki:
    norm = normalize(tfidf_wiki[key])
    tfidf_wiki_norm[key] = norm

for key in tfidf_arxiv:
    norm = normalize(tfidf_arxiv[key])
    tfidf_arxiv_norm[key] = norm
time_norm = timeit.default_timer() - start_time
print("done {}\n".format(time_norm))

# pickle these normalized dictionaries
pickle.dump(tfidf_wiki_norm, 
        open('./data_objects/tfidf_wiki_norm_{}.p'.format(date), 'w'))
pickle.dump(tfidf_arxiv_norm, 
        open('./data_objects/tfidf_arxiv_norm_{}.p'.format(date), 'w'))

# compute the cosine distances
start_time = timeit.default_timer()
i = 0
for key in tfidf_arxiv_norm:
    doc_comparisons[key] = {key2:dot_product(tfidf_arxiv_norm[key],
        tfidf_wiki_norm[key2]) for key2 in tfidf_wiki_norm}
    if (i % 10 == 0): print(i, key)
    i += 1
time_cos = timeit.default_timer() - start_time
print("done {}\n".format(time_cos))

# pickle
pickle.dump(doc_comparisons, 
        open('./data_objects/doc_comparisons_{}.p'.format(date), 'w'))

# write times to file
file_times = open('./data_objects/computation_times_{}.txt'.format(date), 'w')
file_times.write("time_doclist: {}\n".format(time_doclist))
file_times.write("time_vocab: {}\n".format(time_vocab))
files_times.write("time_tfidf_wiki: {}\n".format(time_tfidf_wiki))
files_times.write("time_tfidf_arxiv: {}\n".format(time_tfidf_arxiv))
files_times.write("time_norm: {}\n".format(time_norm))
files_times.write("time_cos: {}\n".format(time_cos))
