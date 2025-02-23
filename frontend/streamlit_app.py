import streamlit as st
from httpx import HTTPStatusError
import sys
import os
import pandas as pd
import spacy
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # import from backend

from backend.faiss_index import (
    load_faiss_index,
    faiss_similarity_check,
    FAISS_SIMILARITY_THRESHOLD,
    update_faiss_index
)
from backend.mistral_api import call_mistral_with_retry
from backend.database.mongodb import insert_prompt_log, get_false_negatives
from backend.hybrid import SIMILARITY_THRESHOLD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TRAIN_CSV = config["train_csv"]

# Neobrutalistic Style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');

    body {
        background-color: #FFD93D;
        color: #333;
        font-family: 'Raleway', sans-serif;
    }
    .stApp {
        padding: 20px;
        background-color: #FFD93D;
    }
    h1 {
        color: #000;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .prompt-container, .stats-container {
        border: 3px solid #000;
        border-radius: 12px;
        background-color: #C084FC;
        padding: 15px;
        box-shadow: 5px 5px 0px #000;
        width: 100%;
        position: relative;
    }
    .stButton button {
        border: 3px solid #000;
        border-radius: 8px;
        background-color: #C084FC;
        color: #000;
        font-weight: 700;
        box-shadow: 4px 4px 0px #000;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #D1A3FF;
        box-shadow: 2px 2px 0px #000;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar 
page = st.sidebar.selectbox("Go to", ["Home", "Statistics", "Feedback Loop"])

if page == "Home":
    st.title("Shick Shack")

    try:
        index = load_faiss_index()
    except FileNotFoundError:
        st.error("FAISS index not found. Please build it first.")
        st.stop()

    st.markdown('<div class="prompt-container"><span class="prompt-label">ENTER YOUR PROMPT:</span>', unsafe_allow_html=True)
    prompt_text = st.text_area("", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Submit"):
        if prompt_text.strip():
            similarity = faiss_similarity_check(prompt_text, index)

            if similarity >= SIMILARITY_THRESHOLD:
                final_label = "Jailbreak Identified in the Primary Phase"
                mistral_resp = "Not required."
            else:
                try:
                    mistral_resp = call_mistral_with_retry(prompt_text)
                except HTTPStatusError as e:
                    if e.response is not None and e.response.status_code == 429:
                        mistral_resp = "SKIPPED_DUE_TO_RATELIMIT"
                        final_label = "Benign"
                        st.warning("Rate-limited by Mistral. Labeling as Benign (skipped).")
                    else:
                        st.error(f"HTTP Error: {e}")
                        st.stop()

                if mistral_resp != "SKIPPED_DUE_TO_RATELIMIT":
                    if "jailbreak" in mistral_resp.lower() and "benign" not in mistral_resp.lower():
                        final_label = "Jailbreak Identified in the Secondary Phase"
                    else:
                        final_label = "Benign"

            try:
                insert_prompt_log(prompt_text, final_label, similarity, mistral_resp)
            except:
                print(f"[MongoDB Logging Error]: {e}")

            st.write(f"**Vector Similarity:** {similarity:.2f} (threshold: {SIMILARITY_THRESHOLD})")
            st.write(f"**Intent Analysis:** {mistral_resp}")
            st.success(f"**Result:** {final_label}")
        else:
            st.warning("Please enter a valid prompt.")

elif page == "Statistics":
    st.title("Training Data Analysis")

    try:
        df = pd.read_csv(TRAIN_CSV)

        nlp = spacy.load("en_core_web_sm")

        # Token Count
        df["token_count"] = df["prompt"].apply(lambda x: len(nlp(x)))

        st.markdown('<div class="stats-container"><h3>Token Count Distribution</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.hist(df["token_count"], bins=50, color="#FF6347")
        ax.set_xlabel("Number of Tokens")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
 
        # TFIDF
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["prompt"])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        top_tfidf = tfidf_df.sum().sort_values(ascending=False).head(10)

        st.markdown('<div class="stats-container"><h3>Top TF-IDF Terms</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        top_tfidf.plot(kind="bar", color="#FF6347", ax=ax)
        ax.set_xlabel("Terms")
        ax.set_ylabel("TF-IDF Score")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Top N-Grams (2 and 3)
        vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words="english")
        ngrams = vectorizer.fit_transform(df["prompt"])
        ngram_freq = pd.DataFrame(ngrams.toarray(), columns=vectorizer.get_feature_names_out())
        top_ngrams = ngram_freq.sum().sort_values(ascending=False).head(10)

        st.markdown('<div class="stats-container"><h3>Top Bigrams & Trigrams</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        top_ngrams.plot(kind="bar", color="#4682B4", ax=ax)
        ax.set_xlabel("N-Grams")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Emotional Quotient
        df["polarity"] = df["prompt"].apply(lambda x: TextBlob(x).sentiment.polarity)
        df["subjectivity"] = df["prompt"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

        st.markdown('<div class="stats-container"><h3>Sentiment Analysis</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.scatter(df.index, df["polarity"], color="#FF6347", label="Polarity", alpha=0.6)
        ax.scatter(df.index, df["subjectivity"], color="#4682B4", label="Subjectivity", alpha=0.6)
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Score")
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Unicode and Markdown 
        def count_unicode(prompt):
            return sum(1 for char in prompt if ord(char) > 127)

        def count_markdown(prompt):
            markdown_patterns = r'(\*\*|\*|`|~~|#|>)'
            return len(re.findall(markdown_patterns, prompt))

        df['unicode_count'] = df['prompt'].apply(count_unicode)
        df['markdown_count'] = df['prompt'].apply(count_markdown)

        result = df.groupby('type').agg({
            'unicode_count': 'sum',
            'markdown_count': 'sum'
        }).reset_index()

        st.markdown('<div class="stats-container"><h3>Unicode and Markdown Usage</h3>', unsafe_allow_html=True)
        st.dataframe(result)
        st.markdown('</div>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Training CSV not found. Please check the config file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif page == "Feedback Loop":
    st.title("Feedback Loop: Improve Primary Phase")

    if st.button("Run Feedback Loop"):
        with st.spinner("Fetching false negatives and updating FAISS index..."):
            false_negatives = get_false_negatives()
            if false_negatives:
                update_faiss_index(false_negatives)
                st.success(f"FAISS index updated with {len(false_negatives)} new prompt(s).")
            else:
                st.info("FAISS index is up-to-date.")
