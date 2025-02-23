import os
import faiss
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FAISS_INDEX_PATH = config["faiss_index_path"]
FAISS_SIMILARITY_THRESHOLD = config["faiss_similarity_threshold"]
FAISS_K = config["faiss_k"]

# Embedding 
embedding_model_name = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embedding_model_name)

def build_faiss_index(jailbreak_texts):
    embeddings = embed_model.encode(jailbreak_texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)  
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[FAISS] Index built with {len(jailbreak_texts)} vectors and saved to {FAISS_INDEX_PATH}.")

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def faiss_similarity_check(prompt, index):
    prompt_embedding = embed_model.encode([prompt]).astype("float32")
    faiss.normalize_L2(prompt_embedding)
    distances, _ = index.search(prompt_embedding, FAISS_K)
    avg_similarity = float(np.mean(distances[0]))
    return avg_similarity
