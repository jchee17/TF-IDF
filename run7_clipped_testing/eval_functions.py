#-*- coding: utf-8 -*-  
from __future__ import unicode_literals
from __future__ import division
import os
import re
import timeit
import codecs
import pickle
import numpy as np

path_uni  = "./data_objects/vocab_uni/"
path_bold = "./data_objects/vocab_bold/"
path_bu   = "./data_objects/vocab_bold_uni/"

# ============================================================================= 
def reorder(path, cos_dist_fname): 
    cos_dist = np.load(path+cos_dist_fname+".npy")
    num_docs = cos_dist.shape[0]
    num_concepts = cos_dist.shape[1]
    reorder = np.zeros((num_docs, num_concepts), dtype=(float,2))

    for i in range(num_docs):
        sorted_indices = np.argsort(cos_dist[i])[::-1]
        reorder[i] = np.array([(idx, cos_dist[i,idx]) for idx in sorted_indices])

    np.save(path+"reorder_"+cos_dist_fname+".npy", reorder)
    #return reorder

def top_N(N, labled_docs_fname, reorder_cos_dist_fname, vocab_dirname,
        list_docs, idx_to_concept):
    """stat1:% of concepts in top_N which are in true labels
       stat2:% of true labels covered by top_N concepts"""
    # tallying statistics
    num_in_N = 0
    tot_N = 0
    tot_label = 0
    #label_cover = 0.0

    # open file pointers
    reorder_cos_dist = np.load("./data_objects/"+vocab_dirname+
            reorder_cos_dist_fname+".npy")
    labled_docs = codecs.open(labled_docs_fname+".txt", 'r', "utf-8", errors="ignore")

    # read lines of labled_docs
    for line in labled_docs.readlines():
        title = line.split('\t',1)[0]
        true_labels = line.split('\t',1)[1].split(',')

        # get top_N concepts
        doc_idx = list_docs.index(title)
        concept_idxs = [reorder_cos_dist[doc_idx, j][0] for j in range(N)]
        concepts = [idx_to_concept[i] for i in concept_idxs]

        # tally
        tot_N += N
        tmp = 0
        for word in concepts:
            if word in true_labels:
                tmp += 1
        num_in_N += tmp
        tot_label += len(true_labels)
        #label_cover += tmp / len(true_labels)

    # write to file
    precision = num_in_N/tot_N
    recall = num_in_N/tot_label
    f1 = (2*precision*recall) / (precision+recall)
    
    outfile = open("./top_n.txt", 'a')
    outfile.write("{}\tN:{}\tprecision:{}\trecall:{}\tf1:{}\tnum_in_N:{}\ttot_N:{}\ttot_label:{}\n".format(reorder_cos_dist_fname, N, precision, recall, f1, num_in_N, tot_N, tot_label))

def top_N_2(N, labled_docs_fname, reorder_cos_dist_fname, vocab_dirname,
        list_docs, idx_to_concept, label_ignore, wiki_ignore):
    """precision:% of concepts in top_N which are in true labels
       recall:% of true labels covered by top_N concepts
       implements ignore lists"""
    # tallying statistics
    num_in_N = 0
    tot_N = 0
    tot_label = 0

    # reverse dict and idx wiki_ignore
    concept_to_idx = {v: k for k, v in idx_to_concept.items()}
    wiki_ignore_idx = [concept_to_idx[ign] for ign in wiki_ignore 
            if ign in concept_to_idx.keys()]

    # open file pointers
    reorder_cos_dist = np.load("./data_objects/"+vocab_dirname+
            reorder_cos_dist_fname+".npy")
    labled_docs = codecs.open(labled_docs_fname+".txt", 'r', "utf-8", errors="ignore")

    # read lines of labled_docs
    for line in labled_docs.readlines():
        title = line.split('\t',1)[0]
        true_labels = line.split('\t',1)[1].split(',')

        # remove label_ignore from true_labels
        set_true_labels = set(true_labels)
        set_label_ignore = set(label_ignore)
        true_labels = list(set_true_labels.difference(set_label_ignore))

        # get top_N concepts
        doc_idx = list_docs.index(title)
        concept_idxs = []
        count = 0
        j = 0
        while (count < N):
            idx = reorder_cos_dist[doc_idx, j][0] 
            # catch if concept in ingore
            if idx not in wiki_ignore_idx:
                concept_idxs.append(idx)
                count += 1
            j += 1
        concepts = [idx_to_concept[i] for i in concept_idxs]

        # tally
        tot_N += N
        tmp = 0
        for word in concepts:
            if word in true_labels:
                tmp += 1
        num_in_N += tmp
        tot_label += len(true_labels)
        #label_cover += tmp / len(true_labels)

    # write to file
    precision = num_in_N/tot_N
    recall = num_in_N/tot_label
    f1 = (2*precision*recall) / (precision+recall)
    
    outfile = open("./top_n.txt", 'a')
    outfile.write("[ignore] {}\tN:{}\tprecision:{}\trecall:{}\tf1:{}\tnum_in_N:{}\ttot_N:{}\ttot_label:{}\n".format(reorder_cos_dist_fname, N, precision, recall, f1, num_in_N, tot_N, tot_label))

def threshold_T(T, labled_docs_fname, reorder_cos_dist_fname, vocab_dirname,
        list_docs, idx_to_concept):
    """stat1:% of concepts in top_N which are in true labels
       stat2:% of true labels covered by top_N concepts"""
    # tallying statistics
    num_in_N = 0
    tot_T = 0
    tot_label = 0

    # open file pointers
    reorder_cos_dist = np.load("./data_objects/"+vocab_dirname+
            reorder_cos_dist_fname+".npy")
    labled_docs = codecs.open(labled_docs_fname+".txt", 'r', "utf-8", errors="ignore")

    # read lines of labled_docs
    for line in labled_docs.readlines():
        title = line.split('\t',1)[0]
        true_labels = line.split('\t',1)[1].split(',')

        # get top_N concepts
        doc_idx = list_docs.index(title)
        #concept_idxs = [reorder_cos_dist[doc_idx, j][0] for j in range(N)]
        concept_idxs = []
        j = 0
        while reorder_cos_dist[doc_idx, j][1] > T:
            concept_idxs.append(reorder_cos_dist[doc_idx, j][0])
            j += 1
        concepts = [idx_to_concept[i] for i in concept_idxs]

        # tally
        tot_T += len(concept_idxs)
        tot_label += len(true_labels)
        for word in concepts:
            if word in true_labels:
                num_in_N += 1
        #label_cover += tmp / len(true_labels)

    # write to file
    precision = num_in_N/tot_T
    recall = num_in_N/tot_label
    f1 = (2*precision*recall) / (precision+recall)
    
    outfile = open("./threshold_t.txt", 'a')
    outfile.write("{}\tT:{}\tprecision:{}\trecall:{}\tf1:{}\tnum_in_N:{}\ttot_T:{}\ttot_label:{}\n".format(reorder_cos_dist_fname, T, precision, recall, f1, num_in_N, tot_T, tot_label))

def threshold_T_2(T, labled_docs_fname, reorder_cos_dist_fname, vocab_dirname,
        list_docs, idx_to_concept, label_ignore, wiki_ignore):
    """stat1:% of concepts in top_N which are in true labels
       stat2:% of true labels covered by top_N concepts"""
    # tallying statistics
    num_in_N = 0
    tot_T = 0
    tot_label = 0

    # reverse dict and idx wiki_ignore
    concept_to_idx = {v: k for k, v in idx_to_concept.items()}
    wiki_ignore_idx = [concept_to_idx[ign] for ign in wiki_ignore
            if ign in concept_to_idx.keys()]

    # open file pointers
    reorder_cos_dist = np.load("./data_objects/"+vocab_dirname+
            reorder_cos_dist_fname+".npy")
    labled_docs = codecs.open(labled_docs_fname+".txt", 'r', "utf-8", errors="ignore")

    # read lines of labled_docs
    for line in labled_docs.readlines():
        title = line.split('\t',1)[0]
        true_labels = line.split('\t',1)[1].split(',')

        # remove label_ignore from true_labels
        set_true_labels = set(true_labels)
        set_label_ignore = set(label_ignore)
        true_labels = list(set_true_labels.difference(set_label_ignore))
        
        # get top_N concepts
        doc_idx = list_docs.index(title)
        concept_idxs = []
        j = 0
        while reorder_cos_dist[doc_idx, j][1] > T:
            idx = reorder_cos_dist[doc_idx, j][0]
            # catch if concept in ignore
            if idx not in wiki_ignore_idx:
                concept_idxs.append(idx)
            j += 1
        concepts = [idx_to_concept[i] for i in concept_idxs]

        # tally
        tot_T += len(concept_idxs)
        tot_label += len(true_labels)
        for word in concepts:
            if word in true_labels:
                num_in_N += 1
        #label_cover += tmp / len(true_labels)

    # write to file
    precision = num_in_N/tot_T
    recall = num_in_N/tot_label
    f1 = (2*precision*recall) / (precision+recall)
    
    outfile = open("./threshold_t.txt", 'a')
    outfile.write("[ignore] {}\tT:{}\tprecision:{}\trecall:{}\tf1:{}\tnum_in_N:{}\ttot_T:{}\ttot_label:{}\n".format(reorder_cos_dist_fname, T, precision, recall, f1, num_in_N, tot_T, tot_label))

