#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: lstm.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 11/27/21 15:53
"""

from math import sqrt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from keras.layers.core import Dense, Dropout
from helper import series_to_supervised, mean_absolute_percentage_error



# load dataset
dataset = read_csv('data/pollution1.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# specify the number of lag hours
n_hours = 3*24
n_features = 8

# frame as supervised learning
reframed = series_to_supervised(values, n_hours, 1)
print("reframed.shape:", reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = int(len(values)*0.7)
print("n_train_hours:", n_train_hours)

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]


# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train_X = scaler.fit_transform(train_X)
train_y = scaler.fit_transform(train_y)
test_X = scaler.fit_transform(test_X)
test_y = scaler.fit_transform(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Simple RNN Model
# lr = 0.0001
# EPOCHS = 150
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.summary()
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y,
                    epochs=50,
                    batch_size=256,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
plt.title("Training loss V.S. Testing loss")
plt.savefig('graph/lstm_loss.png', dpi=300)
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

# invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(yhat)
# inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(test_y)
# inv_y = inv_y[:, 0]
print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)


# calculate RMSE, MAE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = mean_absolute_percentage_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
print('Test MAPE: %.3f' % mape)


# plot Prediction V.S. actual value of PM2.5
# 30609 --> first test data
# 30609 -- 0
# 30626 -- 17 -- 2013-07-01-00:00
# 30985 -- 376 -- 2013-07-15-23:00
plt.plot(inv_yhat[17:376], label='prediction')
plt.plot(inv_y[17:376], label='ground_truth')
plt.title("Prediction V.S. actual value of PM2.5")
plt.legend()
plt.savefig('graph/rnn_prediction.png', dpi=300)
plt.show()
#
#
# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# # model.summary()
# model.compile(loss='mae', optimizer='adam')
#
# # fit network
# history = model.fit(train_X, train_y,
#                     epochs=2,
#                     batch_size=256,
#                     validation_data=(test_X, test_y),
#                     verbose=2,
#                     shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# plt.title("Training loss V.S. Testing loss with LSTM")
# plt.savefig('graph/lstm_loss.png', dpi=300)
# pyplot.show()
#
# # make a prediction
# yhat = model.predict(test_X)
# print(yhat.shape)
# test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
#
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
#
# # invert scaling for actual
# print(test_y.shape)
# test_y = test_y.reshape((len(test_y), 1))
# print(test_y.shape)
# inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
# print(inv_y.shape)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
#
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
#
# # plot Prediction V.S. actual value of PM2.5
# plt.plot(inv_yhat[0:256], label='prediction')
# plt.plot(inv_y[0:256], label='ground_truth')
# plt.title("Prediction V.S. actual value of PM2.5")
# # plt.savefig('graph/LSTM_prediction.png')
# plt.legend()
# plt.show()