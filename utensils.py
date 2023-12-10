import pickle

def load_data(f):
    data = pickle.load(open(f, 'rb'))
    train_x, train_y = data['train_data']
    valid_x, valid_y = data['valid_data']
    test_x, test_y = data['test_data']
    min, max = data['min'], data['max']
    return train_x, train_y, valid_x, valid_y, test_x, test_y, min, max

def scale_back(f, min, max):
    range = max-min
    result = [x*range+min for x in f]
    return result 