import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from app.database import load_model

def preprocess_custom_prompt(prompt, top_tfidf_terms, top_ngrams, tfidf_count_vectorizer, ngram_count_vectorizer):
    # Create a single-row DataFrame for input
    df = pd.DataFrame({'prompt': [prompt]})

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

    # Bigram count
    df['bigram_count'] = [sum(1 for _ in zip(text.split(), text.split()[1:])) for text in df['prompt']]

    # TF-IDF Average
    tfidf_vectorizer_baseline = TfidfVectorizer()
    tfidf_matrix_baseline = tfidf_vectorizer_baseline.fit_transform(df['prompt'])
    df['tfidf_avg'] = tfidf_matrix_baseline.mean(axis=1).A1

    # Frequency for top TF-IDF terms
    tfidf_count_matrix = tfidf_count_vectorizer.transform(df['prompt'])
    tfidf_count_df = pd.DataFrame(tfidf_count_matrix.toarray(), columns=[f"tfidf_{term}" for term in top_tfidf_terms])
    df['total_tfidf_freq'] = tfidf_count_df.sum(axis=1)

    # Frequency for top n-grams
    ngram_count_matrix = ngram_count_vectorizer.transform(df['prompt'])
    ngram_count_df = pd.DataFrame(ngram_count_matrix.toarray(), columns=[f"ngram_{ngram}" for ngram in top_ngrams])
    df['total_ngram_freq'] = ngram_count_df.sum(axis=1)

    return df

# Define a function to predict jailbreak or benign
def predict_custom_prompt(prompt, rf_classifier, top_tfidf_terms, top_ngrams, tfidf_count_vectorizer, ngram_count_vectorizer):
    processed_df = preprocess_custom_prompt(prompt, top_tfidf_terms, top_ngrams, tfidf_count_vectorizer, ngram_count_vectorizer)
    
    # Select only the feature columns used during training
    features = ['unicode_count', 'markdown_count', 'subjectivity', 'char_length', 'token_count', 'bigram_count', 'tfidf_avg', 'total_tfidf_freq', 'total_ngram_freq']
    X_input = processed_df[features]

    # Predict the class
    prediction = rf_classifier.predict(X_input)
    return "Jailbreak" if prediction[0] == 1 else "Benign"
