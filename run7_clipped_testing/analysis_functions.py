#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import re
import pickle
import numpy as np

def label_dist(fname):
    """ collect counts of true labels from labled_docs.txt
        one line per doc, 1st item is title, rest are true labels"""
    f = codecs.open(fname, 'r', "utf-8", errors="ignore")
    counts = {}

    for line in f:
        label_set = line.rstrip('\n').split('\t',1)[1].lower().split(',')
        for label in label_set:
            if label not in counts.keys():
                counts[label] = 1
            else:
                counts[label] += 1

    return counts        

def gen_ignore_words(arr, ls, end):
    """ return list to ignore based on cutoff values """
    cutoff = np.percentile(arr, end)
    print("cutoff: {}".format(cutoff))
    assert(len(arr) == len(ls))
    if end < 50:
        ignore_values = [ ls[i] for i in range(len(arr)) if arr[i] <= cutoff ]
    if end >= 50:
        ignore_values = [ ls[i] for i in range(len(arr)) if arr[i] >= cutoff ]

    return ignore_values

def join_ignore_words(label_ignore, wiki_ignore, wiki_to_concept):
    """ expand label_ignore and wiki_ignore to include each other """
    # reverse dict
    concept_to_wiki = {value:key for key,value in 
            wiki_to_concept.iteritems()}

    # convert input from list to set
    label_ignore = set(label_ignore)
    wiki_ignore  = set(wiki_ignore)

    # trasnform each ignore list to format of other 
    label_transform = set( [concept_to_wiki[ign] for ign in label_ignore 
        if ign in concept_to_wiki.keys()] )
    wiki_transform  = set( [wiki_to_concept[ign] for ign in wiki_ignore 
        if ign in wiki_to_concept.keys()] )

    # expand label and wiki
    label_ignore = list(label_ignore.union(wiki_transform))
    wiki_ignore  = list(wiki_ignore.union(label_transform))

    return (label_ignore, wiki_ignore)

    

