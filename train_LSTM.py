import numpy as np
import tensorflow as tf
import pickle
import os


def load_data(f):
    data = pickle.load(open(f, 'rb'))
    train_x, train_y = data['train_data']
    valid_x, valid_y = data['valid_data']
    test_x, test_y = data['test_data']
    return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    price_only = False
    model_data_path = './model_data/model_data.pkl'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(model_data_path)
    
    if price_only:
        train_x = train_x[:, :, 0]
        valid_x = valid_x[:, :, 0]
        test_x = test_x[:, :, 0]
        train_x = np.reshape(train_x, (len(train_x), 20, 1))
        valid_x = np.reshape(valid_x, (len(valid_x), 20, 1))
        test_x = np.reshape(test_x, (len(test_x), 20, 1))
    
    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape)
    
    
    train_x= tf.convert_to_tensor(train_x, dtype = tf.float32)
    train_y= tf.convert_to_tensor(train_y, dtype = tf.float32)
    valid_x= tf.convert_to_tensor(valid_x, dtype = tf.float32)
    valid_y= tf.convert_to_tensor(valid_y, dtype = tf.float32)
    test_x= tf.convert_to_tensor(test_x, dtype = tf.float32)
    test_y= tf.convert_to_tensor(test_y, dtype = tf.float32)
    
    ### Save the best model Callback
    checkpoint_filepath = './model_data/LSTM'
    
    ### LSTM
    input_dim = 1 if price_only else len(train_x[0][0])
    timestep = len(train_x[0])
    
    ### Grid Search
    rnn_dim = [50, 100, 150, 200]
    hidden_dim = [30, 50, 100, 150]
    for i in rnn_dim:
        for j in hidden_dim:
            save_path = os.path.join(checkpoint_filepath, str(i), str(j), 'model_weight')
            if not save_path:
                os.mkdir(os.path.join(checkpoint_filepath, str(i), str(j), 'model_weight'))
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, 
                                                                    monitor='val_loss', mode='min', save_best_only=True, verbose=1)
            model = tf.keras.Sequential()
        
            model.add(tf.keras.Input(shape = (timestep, input_dim)))
            model.add(tf.keras.layers.LSTM(i))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(j, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
            
            model.compile(optimizer='Adam', loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
            model.fit(train_x, train_y, validation_data = (valid_x, valid_y), batch_size=32, epochs = 200, shuffle=True, callbacks=[model_checkpoint_callback], verbose = 2)
            model.summary()
        
        
            ### Evaluate using the best epoch
            model.load_weights(save_path)
            model.evaluate(test_x, test_y)
    
