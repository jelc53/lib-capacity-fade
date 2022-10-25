def load_data(filename):
    with open(filename, 'rb') as f:
        data = f.load_pickle()
    
    return data

if __name__ == "__main__":
    data_dict = load_data(filename)