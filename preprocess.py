import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler 
from matplotlib import pyplot as plt


### Normalization?
def load_price_data(f):
    df = pd.read_csv(f)
    #df = df[df["Date"]<"2023-01-01"]
    min, max = df['Close'].min(), df['Close'].max()
    df['Close'] = MinMaxScaler().fit_transform(np.array(df['Close']).reshape(-1,1))
    df['Open'] = MinMaxScaler().fit_transform(np.array(df['Open']).reshape(-1,1))
    df['High'] = MinMaxScaler().fit_transform(np.array(df['High']).reshape(-1,1))
    df['Low'] = MinMaxScaler().fit_transform(np.array(df['Low']).reshape(-1,1))
    #df['Volume'] = MinMaxScaler().fit_transform(np.array(df['Volume']).reshape(-1,1))
    data = np.array(df[['Close', 'Open', 'High', 'Low']])
    
    return data, min, max


def build_dataset(data, sequence_length = 21):
    n = len(data)
    n_samples = n-sequence_length+1
    x = np.zeros(shape=(n_samples, sequence_length-1, len(data[0])), dtype=np.float32)
    y = np.zeros(shape=(n_samples, 1), dtype = np.float32)
    for index, i in enumerate(data):
        if index + sequence_length-1 >= n: break
        x[index] = data[index:index+sequence_length-1]
        y[index] = data[index+sequence_length-1][0]
        
    return x, y


### vaild, testing should choose random from the rest
def split_data(x, y):
    n = len(x)
    training, testing = int(0.8*n), int(0.1*n)
    train_x, valid_x, test_x = x[:training], x[training:-testing], x[-testing:]
    train_y, valid_y, test_y = y[:training], y[training:-testing], y[-testing:]
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




if __name__ == "__main__":
    ### Load Dataset, and do normalizations
    data, min, max = load_price_data("./data/MSFT.csv")
    print(data.shape)
    print(f"The minimum price is {min} and the maximum price is {max}")

    ### Build Dataset
    data_x, data_y = build_dataset(data)
    
    ### Split data
    train_data, valid_data, test_data = split_data(data_x, data_y)
    
    
    ### Plot
    plt.plot(data[:, 0])
    plt.axvline(x = len(train_data[1]), color = 'black')
    plt.show()
    
    
    pickle.dump({
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'min': min,
        'max': max
    }, open('./model_data/model_data.pkl', 'wb'))
    
