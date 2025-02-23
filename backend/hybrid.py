import os
import yaml
import pandas as pd
import numpy as np
from httpx import HTTPStatusError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.faiss_index import (
    build_faiss_index, load_faiss_index,
    faiss_similarity_check, update_faiss_index, FAISS_SIMILARITY_THRESHOLD
)
from backend.mistral_api import call_mistral_with_retry
from backend.database.mongodb import insert_prompt_log, get_false_negatives

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TRAIN_CSV = config["train_csv"]
TEST_CSV = config["test_csv"]
FAISS_INDEX_PATH = config["faiss_index_path"]
SIMILARITY_THRESHOLD = config["faiss_similarity_threshold"]
FAISS_K = config["faiss_k"]
USE_MONGO = config["use_mongodb"]

def load_data():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    df_train["label"] = df_train["type"].apply(lambda x: 1 if x == "jailbreak" else 0)
    df_test["label"] = df_test["type"].apply(lambda x: 1 if x == "jailbreak" else 0)
    return df_train, df_test

def build_index_if_needed(df_train):
    if not os.path.exists(FAISS_INDEX_PATH):
        df_jb = df_train[df_train["label"] == 1]
        jb_prompts = df_jb["prompt"].tolist()
        print(f"[Index] Building from {len(jb_prompts)} jailbreak prompts...")
        build_faiss_index(jb_prompts)
    else:
        print(f"[Index] Using existing index at {FAISS_INDEX_PATH}.")

def hybrid_jailbreak_detection(prompt, index):
    avg_similarity = faiss_similarity_check(prompt, index)
    if avg_similarity >= SIMILARITY_THRESHOLD:
        final_label = "Jailbreak"
        mistral_resp = "Skipped (FAISS flagged it)"
    else:
        try:
            mistral_resp = call_mistral_with_retry(prompt)
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"[Rate Limit] Skipping prompt: {prompt}")
                mistral_resp = "SKIPPED_DUE_TO_RATELIMIT"
                final_label = "Benign"
            else:
                raise

        if mistral_resp != "SKIPPED_DUE_TO_RATELIMIT":
            if "jailbreak" in mistral_resp.lower():
                final_label = "Jailbreak"
            else:
                final_label = "Benign"

    insert_prompt_log(prompt, final_label, avg_similarity, mistral_resp)

    return {
        "Prompt": prompt,
        "FAISS_Similarity": avg_similarity,
        "Mistral_Response": mistral_resp,
        "Hybrid_Prediction": final_label
    }

def feedback_loop():
    false_negatives = get_false_negatives()
    if false_negatives:
        print(f"[Feedback] Found {len(false_negatives)} false negatives. Updating FAISS...")
        update_faiss_index(false_negatives)
        print("[Feedback] FAISS index updated.")
    else:
        print("[Feedback] No false negatives found.")

def main():
    df_train, df_test = load_data()
    build_index_if_needed(df_train)
    index = load_faiss_index()

    results = []
    for i, prompt in enumerate(df_test["prompt"]):
        print(f"Evaluating prompt {i+1}/{len(df_test)}")
        outcome = hybrid_jailbreak_detection(prompt, index)
        results.append(outcome)

    df_out = pd.DataFrame(results)
    df_out["pred_label"] = df_out["Hybrid_Prediction"].apply(lambda x: 1 if x == "Jailbreak" else 0)
    df_out["true_label"] = df_test["label"]

    accuracy = accuracy_score(df_out["true_label"], df_out["pred_label"])
    precision = precision_score(df_out["true_label"], df_out["pred_label"])
    recall = recall_score(df_out["true_label"], df_out["pred_label"])
    f1 = f1_score(df_out["true_label"], df_out["pred_label"])

    print("\n=== Hybrid FAISS + Mistral Metrics ===")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    df_out.to_csv("hybrid_faiss_mistral_results.csv", index=False)
    print("[Done] Results saved to 'hybrid_faiss_mistral_results.csv'.")

    # Trigger Feedback Loop
    feedback_loop()

if __name__ == "__main__":
    main()
