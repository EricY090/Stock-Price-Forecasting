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


class Classifier(tf.keras.layers.Layer):
    def __init__(self, hidden_dims, drop_rate, name='classifier'):
        super().__init__(name=name)
        self.dense = tf.keras.layers.Dense(hidden_dims, activation='relu')
        self.dense1 = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        output = self.dense1(output)
        return output


class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, name='attention'):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.u_omega = self.add_weight(name=name + '_u', shape=(attention_dim,), initializer=tf.keras.initializers.GlorotUniform())
        self.w_omega = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(name=self.name + '_w', shape=(hidden_size, self.attention_dim), initializer=tf.keras.initializers.GlorotUniform())

    def call(self, x):
        t = tf.matmul(x, self.w_omega)              ### (None, 20, attention_dim)
        g = tf.tensordot(t, self.u_omega, axes=1)  ### (None, 20)
        alphas = tf.nn.softmax(g)                  ### Attention score
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), axis=-2)
        return output


class RNN(tf.keras.layers.Layer):
    def __init__(self, rnn_dims, attention_dim):
        super().__init__()
        self.rnn_layers = tf.keras.layers.LSTM(rnn_dims, return_sequences=True)
        self.attention = Attention(attention_dim)

    def call(self, input):
        outputs = self.rnn_layers(input)
        outputs= self.attention(outputs)
        return outputs


class Model(tf.keras.Model):
    def __init__(self, rnn_dims, hidden_dims, attention_dim, drop_rate):
        super().__init__()
        self.rnn = RNN(rnn_dims, attention_dim)
        self.classifier = Classifier(hidden_dims, drop_rate)
    
    def call(self, inputs):
        output = self.rnn(inputs)
        output = self.classifier(output)
        return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_data_path = './model_data/model_data.pkl'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(model_data_path)
    
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
    checkpoint_filepath = './model_data/LSTM_Attention'
    
    input_dim = len(train_x[0][0])
    timestep = len(train_x[0])
    rnn_dim = [50, 100, 150, 200]
    hidden_dim = [30, 50, 100, 150]
    attention_dim = 32
    for i in rnn_dim:
        for j in hidden_dim:    
            save_path = os.path.join(checkpoint_filepath, str(i), str(j), 'model_weight')
            if not save_path:
                os.mkdir(os.path.join(checkpoint_filepath, str(i), str(j), 'model_weight'))
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, 
                                                                            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
            
            ### LSTM+Attntion
            model = Model(rnn_dims=i, hidden_dims=j, attention_dim=attention_dim, drop_rate=0.2)
            model.compile(optimizer='Adam', loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
            model.fit(train_x, train_y, validation_data = (valid_x, valid_y), batch_size=32, epochs = 200, shuffle=True, callbacks=[model_checkpoint_callback], verbose = 2)
            model.summary()
            
            ### Evaluate using the best epoch
            model.load_weights(save_path)
            model.evaluate(test_x, test_y)
        
    