#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: lstm.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 11/20/21 13:46
"""
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from helper import series_to_supervised
import matplotlib.pyplot as plt
from pandas import read_csv
from datetime import datetime

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('raw_uci.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution1.csv')


# load dataset pollution.csv
# manually specify column names mark all NA values with 0
# drop the first 24 hours
# load dataset
dataset = read_csv('pollution1.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of lag hours
n_hours = 5*24
n_features = 8

# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
# print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = int(len(values)*0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Simple GRU Model
model = Sequential()
model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.summary()
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

history = model.fit(train_X, train_y,
                    batch_size=256,
                    epochs=50,
                    validation_data=(test_X, test_y))


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title("Training loss V.S. Testing loss with GRU")
plt.legend()
plt.savefig('graph/gru_loss.png', dpi=300)
plt.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# plot Prediction V.S. actual value of PM2.5
# fig = plt.figure()
plt.plot(inv_yhat[0:256], label='prediction')
plt.plot(inv_y[0:256], label='ground_truth')
plt.title("Prediction V.S. actual value of PM2.5")
plt.legend()

plt.savefig('RNN_prediction.png', dpi=300)
plt.show()




# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from keras.optimizers import adam_v2
#
# # %matplotlib inline
#
# df = pd.read_csv('processed.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# # print(df)
#
# # Determine Parameters
# seq_len = 5*24  # observe the data for the past 5 days
# delay = 1*24    # predict the PM2.5 value one day after
#
# df_ = np.array([df.iloc[i : i + seq_len + delay].values for i in range(len(df) - seq_len - delay)])
# # print(df_.shape)
#
#
# np.random.shuffle(df_)
# x = df_[:, :5*24, :]
# y = df_[:, -24, 0]
# # print(x.shape, y.shape)
#
#
# # Split & Normalize the Data
# split = int(y.shape[0]*0.8)
# train_x = x[:split]
# train_y = y[:split]
# test_x = x[split:]
# test_y = y[split:]
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#
#
#
# mean = train_x.mean(axis=0)
# std = train_x.std(axis=0)
# train_x = (train_x - mean) / std
# test_x = (test_x - mean) / std    # Use the mean & std of train. Since there's no way for us to know the future.
#
# lr = 0.0001
# EPOCHS = 150
#
#
# # Simple RNN Model
# model = keras.Sequential()
# model.add(layers.Flatten(input_shape=(120, 11)))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))   # Regression -> No Need for Activation
# model.summary()
# model.compile(optimizer=adam_v2.Adam(learning_rate=lr, decay=lr/EPOCHS),
#               # optimizer='adam',
#               loss='mse',
#               metrics=['mae'])
# history = model.fit(train_x, train_y,
#                     batch_size=256,
#                     epochs=150,
#                     validation_data=(test_x, test_y))
# plt.plot(history.epoch, history.history['mae'], label='train')
# plt.plot(history.epoch, history.history['val_mae'], label='valid')
# plt.title("MSE loss with a simple RNN model")
# plt.legend()
# plt.show()
#
#
# # Evaluation & Prediction (1st batch of test data)
# model.evaluate(test_x, test_y, verbose=2)
#
# test_predict = model.predict(test_x)
# print(test_y.shape, test_predict.shape)
#
# plt.plot(test_predict[0:256], label='prediction')
# plt.plot(test_y[0:256], label='ground_truth')
# plt.title("Prediction V.S. actual value of PM2.5")
# plt.legend()
# plt.show()
#
#
# # # test_predict[:5]
# # # test_data = df[-120:]
# # # test_data = (test_data - mean)/std
# # # test_data
# # #
# # # test_data = np.expand_dims(test_data, axis=0)
# # # test_data.shape
# # #
# # # model.predict(test_data) # 2015.1.1 11pm pM2.5