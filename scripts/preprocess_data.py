"""
This preprocessing function should be used for applying preprocessing to the training dataset.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from textblob import TextBlob

def preprocess_df(df, top_tfidf_terms, top_ngrams, tfidf_count_vectorizer, ngram_count_vectorizer):
    # Unicode count
    df['unicode_count'] = df['prompt'].apply(lambda x: sum(1 for char in x if ord(char) > 127))

    # Markdown count
    markdown_patterns = r'(\*\*|\*|`|~~|#|>)'
    df['markdown_count'] = df['prompt'].apply(lambda x: len(re.findall(markdown_patterns, x)))

    # Character length
    df['char_length'] = df['prompt'].apply(len)

    # Token count
    df['token_count'] = df['prompt'].apply(lambda x: len(x.split()))

    # Subjectivity using TextBlob
    df['subjectivity'] = df['prompt'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # Bigram count (number of two-word combinations)
    df['bigram_count'] = [sum(1 for _ in zip(text.split(), text.split()[1:])) for text in df['prompt']]

    # TF-IDF Average (to keep as a baseline)
    tfidf_vectorizer_baseline = TfidfVectorizer()
    tfidf_matrix_baseline = tfidf_vectorizer_baseline.fit_transform(df['prompt'])
    df['tfidf_avg'] = tfidf_matrix_baseline.mean(axis=1).A1

    # Encode 'type' for classification (benign = 0, jailbreak = 1)
    df['label'] = df['type'].map({'benign': 0, 'jailbreak': 1})

    # Frequency for top TF-IDF terms
    tfidf_count_matrix = tfidf_count_vectorizer.transform(df['prompt'])
    tfidf_count_df = pd.DataFrame(tfidf_count_matrix.toarray(), columns=[f"tfidf_{term}" for term in top_tfidf_terms])
    df['total_tfidf_freq'] = tfidf_count_df.sum(axis=1)

    # Frequency for top n-grams
    ngram_count_matrix = ngram_count_vectorizer.transform(df['prompt'])
    ngram_count_df = pd.DataFrame(ngram_count_matrix.toarray(), columns=[f"ngram_{ngram}" for ngram in top_ngrams])
    df['total_ngram_freq'] = ngram_count_df.sum(axis=1)

    return df