#!/usr/bin/env python

import os
import codecs
import re

impure_doc_dir    = 'arxiv_processed'
concept_name_file = 'concept_names'
clipped_doc_dir = 'clipped_docs'

impure_doc_list  = os.listdir(impure_doc_dir)
concept_names    = [line.rstrip('\n').strip('^ ').lower() for line in open(concept_name_file) if len(line.rstrip('\n')) > 0]

num_files = 0
print "Total Files = {}".format(len(concept_names))
with open('labled_docs.txt', 'w') as outfile:
    for file_name in impure_doc_list:
        if num_files % 100 == 0:
            print "file number: {}".format(num_files)
        num_files += 1
        doc = codecs.open(os.path.join(impure_doc_dir, file_name), 'r', "utf-8", errors="ignore").read().lower()
        doc_concepts = []
        for concept in concept_names:
            try:
                concept = re.sub('^|$', ' ', concept)
                if re.search(concept, doc, re.UNICODE):
                    doc_concepts.append(concept)
                    doc = doc.replace(concept, '')
            except:
                pass
        with open(os.path.join(clipped_doc_dir, file_name), 'w') as clipped_file:
            clipped_file.write(doc.encode('utf-8', errors='ignore'))
        outfile.write(file_name + '\t' + ','.join(doc_concepts) + '\n')
