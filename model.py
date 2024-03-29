'''
This script contains python functions utilised for modelling in main code
Adapted from https://www.kaggle.com/code/equinxx/stock-prediction-gan-twitter-sentiment-analysis/notebook
v1.0
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pickle import dump, load
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import keras
import pickle

# Seed value
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def normalize_data(df, range, target_column):

    '''
    target_column[str]: closing price of stock
    '''

    target_df_series = pd.DataFrame(df[target_column])
    data = pd.DataFrame(df.iloc[:, :])

    X_scaler = MinMaxScaler(feature_range=range)
    y_scaler = MinMaxScaler(feature_range=range)
    X_scaler.fit(data)
    y_scaler.fit(target_df_series)

    X_scale_dataset = X_scaler.fit_transform(data)
    y_scale_dataset = y_scaler.fit_transform(target_df_series)

    dump(X_scaler, open('X_scaler.pkl', 'wb'))
    dump(y_scaler, open('y_scaler.pkl', 'wb'))

    return (X_scale_dataset,y_scale_dataset)

def batch_data(x_data,y_data, batch_size, predict_period):
    X_batched, y_batched, yc = list(), list(), list()

    for i in range(0,len(x_data),1):
        x_value = x_data[i: i + batch_size][:, :]
        y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
        yc_value = y_data[i: i + batch_size][:, :]
        if len(x_value) == batch_size and len(y_value) == predict_period:
            X_batched.append(x_value)
            y_batched.append(y_value)
            yc.append(yc_value)

    return np.array(X_batched), np.array(y_batched), np.array(yc)

def split_train_test(data):
    train_size = len(data) - 100 #80-20 split
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

def predict_index(dataset, X_train, batch_size, prediction_period):
    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[batch_size: X_train.shape[0] + batch_size + prediction_period, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + batch_size:, :].index

    return train_predict_index, test_predict_index


def make_generator_model(input_dim, output_dim, feature_size):
    model = tf.keras.Sequential([LSTM(units = 1024, return_sequences = True,
                                    input_shape=(input_dim, feature_size),recurrent_dropout = 0.3),
                               LSTM(units = 512, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 256, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 128, return_sequences = True, recurrent_dropout = 0.3),
                               LSTM(units = 64, recurrent_dropout = 0.3),
                               Dense(32),
                               Dense(16),
                               Dense(8),
                               Dense(units=output_dim)])
    return model

def make_discriminator_model(input_dim):
    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(8, input_shape=(input_dim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    #cnn_net.add(Flatten())
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation='sigmoid'))

    return cnn_net

def discriminator_loss(real_output, fake_output):
    loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_f(tf.ones_like(real_output), real_output)
    fake_loss = loss_f(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = loss_f(tf.ones_like(fake_output), fake_output)
    return loss

@tf.function
def train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(real_x, training=True)
        generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
        d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
        real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
        d_real_input = tf.concat([real_y_reshape, yc], axis=1)

        real_output = discriminator(d_real_input, training=True)
        fake_output = discriminator(d_fake_input, training=True)

        g_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': g_loss}

def train(stock_name, real_x, real_y, yc, Epochs, generator, discriminator, g_optimizer, d_optimizer, checkpoint = 50):
    train_info = {}
    train_info["discriminator_loss"] = []
    train_info["generator_loss"] = []

    for epoch in tqdm(range(Epochs)):
        real_price, fake_price, loss = train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer)
        G_losses = []
        D_losses = []
        Real_price = []
        Predicted_price = []
        D_losses.append(loss['d_loss'].numpy())
        G_losses.append(loss['g_loss'].numpy())
        Predicted_price.append(fake_price.numpy())
        Real_price.append(real_price.numpy())

        #Save model every X checkpoints
        if (epoch + 1) % checkpoint == 0:
            # tf.keras.models.save_model(generator, f'./models_gan/{stock_name}/generator_V_%d.h5' % epoch)
            # tf.keras.models.save_model(discriminator, f'./models_gan/{stock_name}/discriminator_V_%d.h5' % epoch)
            tf.keras.models.save_model(generator, f'./models_gan/{stock_name}/generator_V_%d.keras' % epoch)
            tf.keras.models.save_model(discriminator, f'./models_gan/{stock_name}/discriminator_V_%d.keras' % epoch)
            print('epoch', epoch + 1, 'discriminator_loss', loss['d_loss'].numpy(), 'generator_loss', loss['g_loss'].numpy())

        train_info["discriminator_loss"].append(D_losses)
        train_info["generator_loss"].append(G_losses)

    Predicted_price = np.array(Predicted_price)
    Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
    Real_price = np.array(Real_price)
    Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

    plt.subplot(2,1,1)
    plt.plot(train_info["discriminator_loss"], label='Disc_loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_info["generator_loss"], label='Gen_loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()

    plt.show()

    return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(Real_price)

def plot_results(Real_price, Predicted_price, index_train):
    X_scaler = load(open('/content/X_scaler.pkl', 'rb'))
    y_scaler = load(open('/content/y_scaler.pkl', 'rb'))
    train_predict_index = index_train

    rescaled_Real_price = y_scaler.inverse_transform(Real_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=train_predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=train_predict_index[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"])
    plt.plot(predict_result["predicted_mean"], color = 'r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("The result of Training", fontsize=20)
    plt.show()

    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)

## Test Code
@tf.function
def eval_op(generator, real_x):
    generated_data = generator(real_x, training = False)
    return generated_data

def transform_scaler(Real_type_price, Predicted_type_price, index_type, output_dim):
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    predict_index = index_type

    rescaled_Real_price = y_scaler.inverse_transform(Real_type_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_type_price)

    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=predict_index[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    return predict_result, real_price

def plot_data(Real_type_price, Predicted_type_price, index_type, output_dim, stock_name, type):
    predict_result, real_price = transform_scaler(Real_type_price, Predicted_type_price, index_type, output_dim)

    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    MAPE = mean_absolute_percentage_error(predicted, real)
    print(f'{type} RMSE: ', RMSE)
    print(f'{type} MAPE: ', MAPE)

    plt.figure(figsize=(20, 5))
    plt.plot(real_price["real_mean"], color='#00008B')
    plt.plot(predict_result["predicted_mean"], color = '#8B0000', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title(f"Prediction on {type} data for {stock_name}", fontsize=20)
    plt.show()

def prep_data_run_model(stock_name, final, learning_rate, epochs, batch_size, predict_period, SA_indicator):
    X_scale_dataset,y_scale_dataset = normalize_data(final, (-1,1), "Close_es3")
    X_batched, y_batched, yc = batch_data(X_scale_dataset, y_scale_dataset, batch_size = batch_size, predict_period = predict_period)
    print("X shape:", X_batched.shape)
    print("y shape:", y_batched.shape)
    print("yc shape:", yc.shape)

    X_train, X_test, = split_train_test(X_batched)
    y_train, y_test, = split_train_test(y_batched)
    yc_train, yc_test, = split_train_test(yc)
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    if SA_indicator==0:
        # Dataset without sentiment analysis
        X_train = X_train[:,:,:-3]
        X_test = X_test[:,:,:-3]

    index_train, index_test, = predict_index(final, X_train, batch_size, predict_period)
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    g_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    generator = make_generator_model(X_train.shape[1], output_dim, X_train.shape[2])
    discriminator = make_discriminator_model(X_train.shape[1])
    predict_price, real_price, RMSPE = train(stock_name, X_train, y_train, yc_train, epochs, generator, discriminator, g_optimizer, d_optimizer)
    test_generator = tf.keras.models.load_model(f'./models_gan/{stock_name}/generator_V_{epochs-1}.keras', compile=False)
    predict_test_data = eval_op(test_generator, X_test)
    plot_data(y_train, predict_price, index_train, output_dim, stock_name, 'train')
    plot_data(y_test, predict_test_data,index_test, output_dim, stock_name, 'test')

    save_np_output(X_train, 'X_train' + f'_b{batch_size}', stock_name)
    save_np_output(y_train, 'y_train' + f'_b{batch_size}', stock_name)
    save_np_output(yc_train, 'yc_train' + f'_b{batch_size}', stock_name)
    save_np_output(X_test, 'X_test' + f'_b{batch_size}', stock_name)
    save_np_output(y_test, 'y_test' + f'_b{batch_size}', stock_name)
    save_np_output(yc_test, 'yc_test' + f'_b{batch_size}', stock_name)
    save_np_output(index_train, 'index_train' + f'_b{batch_size}', stock_name)
    save_np_output(index_test, 'index_test' + f'_b{batch_size}', stock_name)
    save_np_output(output_dim, 'output_dim' + f'_b{batch_size}', stock_name)
    save_np_output(index_train, 'index_train' + f'_b{batch_size}', stock_name)
    save_np_output(index_test, 'index_test' + f'_b{batch_size}', stock_name)
    save_np_output(output_dim, 'output_dim' + f'_b{batch_size}', stock_name)
    save_np_output(predict_price, 'predict_price' + f'_b{batch_size}', stock_name)
    save_np_output(real_price, 'real_price' + f'_b{batch_size}', stock_name)
    save_np_output(predict_test_data, 'predict_test_data' + f'_b{batch_size}', stock_name)

    # return X_train, y_train, yc_train, X_test, y_test, yc_test, index_train, index_test, output_dim, predict_price, real_price, predict_test_data

def returns_strat(type, batch_size, stock_name):
    #load all relevant datasets
    output_dim = load_np_output('output_dim' + f'_b{batch_size}', stock_name)
    if type=='Test':
        y_test = load_np_output('y_test' + f'_b{batch_size}', stock_name)
        predict_test_data = load_np_output('predict_test_data' + f'_b{batch_size}', stock_name)
        index_test = load_np_output('index_test' + f'_b{batch_size}', stock_name)
        pred, real = transform_scaler(y_test, predict_test_data, index_test, output_dim)
    elif type=='Train':
        y_train = load_np_output('y_train' + f'_b{batch_size}', stock_name)
        predict_train_data = load_np_output('predict_price' + f'_b{batch_size}', stock_name)
        index_train = load_np_output('index_train' + f'_b{batch_size}', stock_name)
        pred, real = transform_scaler(y_train, predict_train_data, index_train, output_dim)

    ret_pred = pred[['predicted_mean']].pct_change()
    ret_real = real[['real_mean']].pct_change()
    ret_pred['position'] = ret_pred.apply(np.sign)
    ret_pred = ret_pred.join(ret_real)
    ret_pred['strat_ret'] = ret_pred['position'].shift(1) * ret_pred['real_mean']
    returns = ret_pred.reset_index().rename(columns={'index':'Date'})

    returns["CumRet_1d_rebalance"] = (
      returns["strat_ret"].expanding().apply(lambda x: np.prod(1 + x) - 1)
    )
    returns["Buy_and_hold_Ret"] = (
      returns["real_mean"].expanding().apply(lambda x: np.prod(1 + x) - 1)
    )
    Final_Return = np.prod(1 + returns["strat_ret"]) - 1
    Buy_Return = np.prod(1 + returns["real_mean"]) - 1

    print("Strategy daily rebalance Return: %.2f %%" % (Final_Return * 100))
    print("Buy and Hold Return: %.2f %%" % (Buy_Return * 100))

    ax = plt.gca()
    returns.plot(x="Date", y="Buy_and_hold_Ret", ax=ax)
    returns.plot(x="Date", y="CumRet_1d_rebalance", ax=ax)
    plt.title(f'{type}: Returns wrt Trading Strategies')
    plt.show()
    return Final_Return, Buy_Return, returns

def save_np_output(df, filename, stock_name):
    np.save(f'./models_gan/{stock_name}/{filename}', df)

def load_np_output(filename, stock_name):
    return np.load(f'./models_gan/{stock_name}/{filename}.npy')
