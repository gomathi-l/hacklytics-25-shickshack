import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="My ML Web App",
    description="A FastAPI backend for ML model inference and retraining.",
    version="1.0",
    debug=True  # Enables debugging mode
)


# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI is running in debug mode!"}

# Run the app when executing main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
