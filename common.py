import pickle


def load_data(filename):
    """Load generic pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data
