import os
import yaml
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "..", "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

USE_MONGODB = config["use_mongodb"]
if USE_MONGODB:
    MONGODB_URI = config["mongodb_uri"]
    DB_NAME = config["database_name"]
    COLLECTION_NAME = config["collection_name"]
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
else:
    client = None
    collection = None

def insert_prompt_log(prompt, classification, similarity, mistral_response=None):
    if collection is None:
        return
    doc = {
        "prompt": prompt,
        "classification": classification,
        "similarity": similarity,
        "mistral_response": mistral_response,
    }
    collection.insert_one(doc)

def get_false_negatives():
    if collection is None:
        print("[MongoDB] No collection initialized.")
        return []

    # Assume FAISS flags Benign if similarity < 0.5
    FAISS_SIMILARITY_THRESHOLD = 0.5

    # Find prompts where FAISS says Benign but Mistral flagged Jailbreak
    false_negatives = list(collection.find({
        "similarity": { "$lt": FAISS_SIMILARITY_THRESHOLD },
        "mistral_response": { "$regex": "jailbreak", "$options": "i" }
    }))

    print(f"[DEBUG] Found {len(false_negatives)} false negatives in MongoDB.")

    for doc in false_negatives:
        print(f"[DEBUG] Prompt: {doc['prompt']} | Mistral Response: {doc['mistral_response']}")

    return [doc["prompt"] for doc in false_negatives]
