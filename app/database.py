import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import pickle

# Load environment variables
load_dotenv()

# MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["ml_database"]  # Name of the database

# Define your collections
training_data_collection = db["training_data"]  # Collection for features and labels
predictions_collection = db["predictions"]      # Optional: Store predictions

def get_training_data():
    """
    Retrieve training data from MongoDB and return it as a Pandas DataFrame.
    """
    data = list(training_data_collection.find())
    if data:
        # Convert MongoDB documents to a Pandas DataFrame
        df = pd.DataFrame(data)
        # Drop MongoDB _id field if it exists
        df = df.drop(columns=["_id"], errors="ignore")
        return df
    return pd.DataFrame()  # Return empty DataFrame if no data is found

def store_training_data(data: pd.DataFrame):
    """
    Store new training data in MongoDB.
    """
    # Convert DataFrame to dictionary and insert into MongoDB
    data_dict = data.to_dict(orient="records")
    training_data_collection.insert_many(data_dict)

def append_to_training_data(new_data: pd.DataFrame):
    """
    Append new data to the existing training dataset in MongoDB.
    """
    # Convert new data to dictionary format
    new_data_dict = new_data.to_dict(orient="records")
    # Insert the new data into the collection
    training_data_collection.insert_many(new_data_dict)

def store_prediction(prediction_data: dict):
    """
    Store model predictions in the MongoDB predictions collection.
    """
    predictions_collection.insert_one(prediction_data)

def load_model(model_name: str):
    """
    Load a trained model from the file system.
    """
    model_path = os.path.join(os.path.dirname(__file__), f"../data/models/{model_name}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
