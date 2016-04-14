def compute_conditional_cos(cos_dist, doc_term_articles, title_concepts):
    """ sets cos_dist score to zero if title of a concept 
        not present in the article """
    num_docs = cos_dist.shape[0]
    num_concepts = cos_dist.shape[1]
    num_vocab = doc_term_articles.shape[1]
    
    cond_cos_dist = np.copy(cos_dist)
    
    titleIdx = 0
    for j in range(num_concepts):
        titleIdx = title_concept[j]
        
        for i in range(num_docs):
            if doc_term_articles[i, titleIdx] == 0:
                # set cond count to 0
                cond_cos_dist[i,j] = 0
        
    return cond_cos_dist

            
