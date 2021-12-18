#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: lstm_multistep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/14/21 10:51
"""


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from helper import series_to_supervised
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout
from pandas import read_csv
from datetime import datetime


# load dataset
dataset = read_csv('data/pollution1.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)

# specify the number of lag hours
n_hours = 3*24
n_features = 8

# frame as supervised learning
reframed = series_to_supervised(values, n_hours, 3)
# print(reframed.shape)

# split into train and test sets
reframed_values = reframed.values
n_train_hours = int(len(reframed_values)*0.7)
train = reframed_values[:n_train_hours, :]
test = reframed_values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, [-n_features*3, -n_features*2, -n_features]]
test_X, test_y = test[:, :n_obs], test[:, [-n_features*3, -n_features*2, -n_features]]

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train_X = scaler.fit_transform(train_X)
train_y = scaler.fit_transform(train_y)
test_X = scaler.fit_transform(test_X)
test_y = scaler.fit_transform(test_y)
print("train_X.shape, train_y.shape:", train_X.shape, train_y.shape)



# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape:\n", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(train_y.shape[1]))
# model.summary()
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y,
                    epochs=10,
                    batch_size=256,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
plt.title("Training loss V.S. Testing loss with LSTM")
# plt.savefig('./graph/lstm_loss.png', dpi=300)
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
print("yhat.shape:", yhat.shape)

test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
print("test_X.shape:", test_X.shape)

# invert scaling for forecast
# inv_yhat = concatenate((yhat[:,0], test_X[:, -7:]), axis=1)    # -7 means except the first column - PM2.5
# inv_yhat = scaler.inverse_transform(yhat)
# inv_yhat = inv_yhat[:, 0]
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = inv_yhat.reshape((-1, 1))

# invert scaling for actual
print("test_y.shape", test_y.shape)
test_y = test_y.reshape((len(test_y), 3))
print("test_y.shape", test_y.shape)
# inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

inv_y = scaler.inverse_transform(test_y)
inv_y = inv_y.reshape((-1, 1))

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

inv_yhat = inv_yhat.reshape((-1, 3))
inv_y = inv_y.reshape((-1, 3))
print("inv_yhat.shape:", inv_yhat.shape)
print("inv_y.shape:", inv_y.shape)

# # plot Prediction V.S. actual value of PM2.5
# plt.plot(inv_yhat[0:256, 0], label='T1_prediction')
# plt.plot(inv_yhat[0:256, 1], label='T2_prediction')
# plt.plot(inv_yhat[0:256, 2], label='T3_prediction')
# plt.plot(inv_y[0:256, 0], label='ground_truth')
#
# plt.title("Predicted V.S. Actual Value")
# plt.xlabel("Timestamps")
# plt.ylabel("PM2.5")
# # plt.savefig('./graph/LSTM_prediction.png')
# plt.legend()
# plt.show()



# show the difference
T1_diff = abs(inv_yhat[0:256, 0]-inv_y[0:256, 0])
T2_diff = abs(inv_yhat[0:256, 1]-inv_y[0:256, 1])
T3_diff = abs(inv_yhat[0:256, 2]-inv_y[0:256, 2])
plt.plot(T1_diff, label='T1_difference')
plt.plot(T2_diff, label='T2_difference')
plt.plot(T3_diff, label='T3_difference')
# plt.plot(inv_y[0:256, 0], label='ground_truth')

plt.title("Differences between Predicted and Actual Value")
plt.xlabel("Timestamps")
plt.ylabel("Difference Value")
# plt.savefig('./graph/RNN_prediction.png')
plt.legend()
plt.show()