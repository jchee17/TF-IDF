# TF-IDF
Information Retrieval search by cosine-distance ranking of document vector 
representations. 
The document vector representations are based on the
term frequency - inverse document frequency (TF-IDF) score, per word,
per document.

Given a scientific paper, determine the most relevant wikipedia articles.
The scientific paper is the 'query', the returned ranked wikipedia aritlces
are the search results. 
The purpose of this project was to determine the feasibility of applying
topic modeling by LDA. 
This information retrieval search can be viewed as a rudimentary topic 
decomposition of the statistics papers, by statistical and mathematical
wikipedia articles.
This project is a step towards the Hopper Project's overall goal of
developing statistical machine learning algorithms for extracting 
representations of text and mathematical expressions and building new 
search tools for the published scientific literature

The code takes in the paths to two directories, one for the corpus of 
scientific papers, and the other for the corpus of wikipedia articles.
It first determines a common vocabulary between the two corpora
(unigrams, bi-grams, frequency cutoffs, etc). 
Next, document-term matrices are constructed per corpus. 
Each row is a document, each column a member of the vocabulary. In the cells 
are the frequency counts. 
From the document-term matrix, document-TFIDF matrices are constructed.
For v in vocabulary and document d in a corpus,

TFIDF(v,d) = TF(v,d) * IDF(v,d)

where TF(v,d) = (count of v in d)/(number of vocab terms in d)
and  IDF(v,d) = log(number of documents in a corpus/number of documents 
which contain v)

We now have two document-TFIDF matrices, one for the corpus of scientific
papers, the other for the corpus of wikipedia articles. 
A final matrix is constructed from these two document-TFIDF corpora,
the cosine distance between the TFIDF vector representation of each
scientific paper (rows) and each wikipedia article (columns).

To perform search over the wikipedia articles given a scientific paper
query, find the corresponding row in the matrix and then order the 
wikipedia articles by their cosine distance scores.
 
 
