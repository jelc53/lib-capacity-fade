import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

if __name__ == "__main__":
    data_dict = load_data(filename)
