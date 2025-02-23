import streamlit as st
from httpx import HTTPStatusError
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # import from backend

from backend.faiss_index import (
    load_faiss_index,
    faiss_similarity_check,
    FAISS_SIMILARITY_THRESHOLD
)
from backend.mistral_api import call_mistral_with_retry
from backend.database.mongodb import insert_prompt_log  # logs
from backend.hybrid import SIMILARITY_THRESHOLD, FAISS_INDEX_PATH

# Apply corrected neobrutalistic style with updated fonts
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
        box-shadow: none;
        background-color: #FFD93D;
    }
    h1 {
        color: #000;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .prompt-container {
        border: 3px solid #000;
        border-radius: 12px;
        background-color: #C084FC;
        padding: 15px;
        box-shadow: 5px 5px 0px #000;
        width: 100%;
        position: relative;
    }
    .prompt-label {
        color: #000;
        font-weight: 700;
        font-size: 1.2em;
        display: block;
        margin-bottom: 5px;
    }
    .stTextArea textarea {
        border: none;
        border-radius: 8px;
        background-color: #FFFFFF;
        font-family: 'Raleway', sans-serif;
        width: 100%;
        height: 150px;
        color: #000;
        resize: none;
        padding: 10px;
        box-shadow: 3px 3px 0px #000;
        font-weight: 400;
    }
    .stButton button {
        border: 3px solid #000;
        border-radius: 8px;
        background-color: #C084FC;
        color: #000;
        font-weight: 700;
        box-shadow: 4px 4px 0px #000;
        padding: 10px 20px;
        font-family: 'Raleway', sans-serif;
    }
    .stButton button:hover {
        background-color: #D1A3FF;
        box-shadow: 2px 2px 0px #000;
    }
    .stSuccess, .stError, .stWarning {
        border: 3px solid #000;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 5px 5px 0px #000;
        background-color: #FFFFFF;
        font-family: 'Raleway', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Shick Shack")

# FAISS Index
try:
    index = load_faiss_index()
except FileNotFoundError:
    st.error("FAISS index not found. Please build it first.")
    st.stop()

st.markdown('<div class="prompt-container"><span class="prompt-label">ENTER YOUR PROMPT:</span>', unsafe_allow_html=True)

prompt_text = st.text_area("", height=150)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze"):
    if prompt_text.strip():
        # Step A: FAISS similarity check
        similarity = faiss_similarity_check(prompt_text, index)

        if similarity >= SIMILARITY_THRESHOLD:
            final_label = "Jailbreak Identified in the Primary Phase"
            mistral_resp = "Not required."
        else:
            # Step B: Call Mistral (secondary)
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
            else:
                final_label = "Benign"

        # Insert log into Mongo
        try:
            insert_prompt_log(prompt_text, final_label, similarity, mistral_resp)
        except:
            pass

        st.write(f"**Vector Similarity:** {similarity:.2f} (threshold: {SIMILARITY_THRESHOLD})")
        st.write(f"**Intent Analysis:** {mistral_resp}")
        st.success(f"**Result:** {final_label}")

    else:
        st.warning("Please enter a valid prompt.")
