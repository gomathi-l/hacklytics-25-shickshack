import pickle

def load_model(model):
    """
    Load the pre-trained ML model.
    """
    with open(model, "rb") as f:
        model = pickle.load(f)
    return model