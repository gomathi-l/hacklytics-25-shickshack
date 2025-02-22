from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_top_tfidf_terms(tfidf_vectorizer, n=50):
    # Get the vocabulary and sort it by term frequency
    terms = tfidf_vectorizer.get_feature_names_out()
    sorted_terms = sorted(zip(tfidf_vectorizer.idf_, terms), key=lambda x: x[0])
    
    # Return the top n terms based on the IDF (Inverse Document Frequency)
    return [term for _, term in sorted_terms[:n]]

def get_top_ngrams(ngram_vectorizer, n=30):
    # Get the vocabulary and sort it by n-gram frequency
    ngrams = ngram_vectorizer.get_feature_names_out()
    return ngrams[:n]  # Here we are just returning the first n n-grams
