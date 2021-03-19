import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from enum import Enum
from pytalib.indicators import trend
from pytalib.indicators import base


class Cell(Enum):
    BasicRNN = 1
    BasicLSTM = 2
    LSTMCellPeephole = 3
    GRU = 4


valid_set_size_percentage = 10
test_set_size_percentage = 10

df = pd.read_csv('data_2019-01-06.csv')
df.sort_values('Date')


# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df['Open'].values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df['High'].values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    df['Volume'] = min_max_scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    return df


# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


# show predictions: 0 = open, 1 = close, 2 = highest, 3 = lowest, 4 = volume
def show_predictions(ft, y_test_pred):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    plt.plot(np.arange(y_test.shape[0]),
             y_test[:, ft], color='black', label='test target')

    plt.plot(np.arange(y_test_pred.shape[0]),
             y_test_pred[:, ft], color='green', label='test prediction')

    plt.title('future stock prices')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best')

    x = 0
    error_percent = 5
    for index in range(0, len(y_test)):
        if (abs((y_test_pred[:, ft][index] - y_test[:, ft][index])) / abs(y_test[:, ft][index]) * 100) < error_percent:
            x += 1
    print("Percent of predictions which error is less then {}% = {}%".format(error_percent, x / len(y_test) * 100))

    # Calculating the direction between 2 points using true values and predicted values
    z = 0
    distance = 10
    for index in range(distance, len(y_test)):
        if (y_test[:, ft][index - distance] <= y_test[:, ft][index] and y_test_pred[:, ft][index - distance] <=
            y_test_pred[:, ft]
            [index]) or (
                y_test[:, ft][index - distance] >= y_test[:, ft][index] and y_test_pred[:, ft][index - distance]
                >= y_test_pred[:, ft][index]):
            z += 1
    print("Percent of correct predicted direction = {}%".format(z / len(y_test) * 100))

    plt.show()


# choose one stock
df_stock = df.copy()
df_stock.drop(['Date'], 1, inplace=True)
cols = list(df_stock.columns.values)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 50  # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)

index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)


# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# parameters
CellType = Cell.BasicRNN
n_steps = seq_len - 1
n_inputs = 5
n_neurons = 200
n_outputs = 5
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 10
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

if CellType == Cell.BasicRNN:
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
              for layer in range(n_layers)]
elif CellType == Cell.BasicLSTM:
    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
              for layer in range(n_layers)]
elif CellType == Cell.LSTMCellPeephole:
    layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
                                      activation=tf.nn.leaky_relu, use_peepholes=True)
              for layer in range(n_layers)]
elif CellType == Cell.GRU:
    layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
              for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:, n_steps - 1, :]  # keep only last output of sequence

loss = tf.reduce_mean(tf.square(outputs - y))  # loss function = mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


# run graph
def train_data(model_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for iteration in range(int(n_epochs * train_set_size / batch_size)):
            x_batch, y_batch = get_next_batch(batch_size)  # fetch the next training batch
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if iteration % int(5 * train_set_size / batch_size) == 0:
                mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
                mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
                print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                    iteration * batch_size / train_set_size, mse_train, mse_valid))
        saver.save(sess, 'train_models/' + model_name)


def test(model_name):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'train_models/' + model_name)
        y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    show_predictions(1, y_test_pred)


model = ['train_model', 'train_model_LSTM', 'train_model_with_batch_500', 'train_model_with_layers_4',
         'train_model_with_volume', 'model_seq_len_100', "model_GRU", 'model_LSTM_pipehole']

# train_data(model[0])
# test(model[0])
y_new = []
for i in y_test:
    y_new.append(i[1] * 10000)
macd = trend.MovingAverageConvergenceDivergence(y_new)
print(macd.calculate())
macd.validate()

tt = trend.ExponentialMovingAverage(y_new, 10)
print(tt.calculate())

plt.figure(figsize=(15, 5))
plt.subplot(1, 1, 1)
plt.plot(np.arange(len(y_new)), y_new, color='black', label='test target')
plt.plot(np.arange(len(macd.macd)), tt.calculate(), color='green', label='test prediction')
plt.plot(np.arange(len(macd.macd)), macd.macd_signal_line, color='red', label='test prediction')
plt.show()
