import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from app.preprocessing import predict_custom_prompt
from app.database import load_model
from scripts.vectorizer import get_top_tfidf_terms, get_top_ngrams
from pydantic import BaseModel

# Request Model
class PromptRequest(BaseModel):
    prompt: str

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Soapify",
    description="Hacklytics 2025; Web app for prompt sanitization",
    version="1.0",
    debug=True  # Enables debugging mode
)

# Load models and preprocessing artifacts
rf_classifier = load_model("random_forest_model.pkl")
tfidf_count_vectorizer = load_model("tfidf_vectorizer.pkl")
ngram_count_vectorizer = load_model("ngram_vectorizer.pkl")

# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI is running in debug mode!"}

# Endpoint: Process User Prompt
@app.post("/predict/")
async def predict(request: PromptRequest):
    # Predict classification for the user prompt
    prediction = predict_custom_prompt(
        request.prompt,
        rf_classifier,
        get_top_tfidf_terms(tfidf_count_vectorizer),
        get_top_ngrams(ngram_count_vectorizer),
        tfidf_count_vectorizer,
        ngram_count_vectorizer
    )
    return {"prompt": request.prompt, "prediction": prediction}

# Run the app when executing main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
