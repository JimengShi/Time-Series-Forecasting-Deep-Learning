#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: rnn_multistep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/14/21 15:46
"""

from matplotlib import pyplot
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from helper import series_to_supervised, mean_absolute_percentage_error

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
n_hours = 4*24
n_features = 8

# frame as supervised learning
reframed = series_to_supervised(values, n_hours, 3)
# print(reframed.shape)

# split into train and test sets
reframed_values = reframed.values
n_train_hours = int(len(reframed_values)*0.6998)  # 0.7 = 0.6998
train = reframed_values[:n_train_hours, :]
test = reframed_values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, [-n_features*3, -n_features*2, -n_features]]
test_X, test_y = test[:, :n_obs], test[:, [-n_features*3, -n_features*2, -n_features]]
truth_y = test[:, -n_features*4]
# test_X1, test_y1 = test_X, test_y
# print("test_X1.shape, test_y1.shape:", test_X1.shape, test_y1.shape)

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
lr = 0.00001
EPOCHS = 200
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(3))   # Regression -> No Need for Activation
model.summary()
model.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
              # optimizer='adam',
              loss='mse',
              metrics=['mae'])
# fit network
history = model.fit(train_X, train_y,
                    epochs=EPOCHS,
                    batch_size=512,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
plt.title("Training loss V.S. Testing loss")
# plt.savefig('./graph/rnn_loss.png', dpi=300)
pyplot.show()
pyplot.close()

# make a prediction
yhat = model.predict(test_X)
print("yhat.shape:", yhat.shape)

test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
print("test_X.shape:", test_X.shape)

# invert scaling for forecast and actual
inv_yhat = scaler.inverse_transform(yhat)
test_y = test_y.reshape((len(test_y), 3))
# print("test_y.shape", test_y.shape)
inv_y = scaler.inverse_transform(test_y)

inv_yhat = inv_yhat.reshape((-1, 1))
inv_y = inv_y.reshape((-1, 1))


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = mean_absolute_percentage_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
print('Test MAPE: %.3f' % mape)


inv_yhat = inv_yhat.reshape((-1, 3))
inv_y = inv_y.reshape((-1, 3))

dates = ['07-04', '07-07', '07-10', '07-13', '07-16', '07-19']

# plot Prediction V.S. actual value of PM2.5
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.plot(inv_yhat[20:379, 0], label='T1_prediction')
plt.plot(inv_yhat[20:379, 1], label='T2_prediction')
plt.plot(inv_yhat[20:379, 2], label='T3_prediction')
plt.plot(inv_y[20:379, 0], label='ground_truth')
plt.xlabel("Time", fontsize='14')
plt.ylabel("PM2.5", fontsize='14')
plt.xticks(np.arange(0, 361, 72), dates, fontsize=14)
plt.yticks(fontsize=14)
plt.title('Predicted (T1, T2, T3) V.S. Actual Value of PM2.5', fontsize=16)
plt.legend(prop={"size": 12}, loc='upper right')
plt.show()
plt.close()


# show the difference
plt.plot(np.subtract(inv_yhat[20:379, 0], inv_y[20:379, 0]), label='T1_difference')     # truth_y[10:369]
plt.plot(np.subtract(inv_yhat[20:379, 1], inv_y[20:379, 0]), label='T2_difference')
plt.plot(np.subtract(inv_yhat[20:379, 2], inv_y[20:379, 0]), label='T3_difference')
# plt.plot(inv_y[0:256, 0], label='ground_truth')
plt.xticks(np.arange(0, 361, 72), dates, fontsize=14)
plt.yticks(fontsize=14)
plt.title("Difference between Predicted and Actual Value of PM2.5", fontsize=16)
plt.xlabel("Time", fontsize='14')
plt.ylabel("Difference", fontsize='14')
# plt.savefig('./graph/RNN_prediction.png')
plt.legend(prop={"size": 12}, loc='upper right')
plt.show()
plt.close()




# # (3, 1) subplot
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1.plot(inv_yhat[17:376, 0], label='T1_prediction')
# ax1.plot(inv_y[17:376, 0], label='T1_truth')
# ax1.set_title("Predicted V.S. Actual Value of T1, T2, T3", fontsize='16')
# ax1.set_xlabel("Timestamps", fontsize='14')
# ax1.set_ylabel("PM2.5 (T1)", fontsize='14')
# ax1.set_xticks([0, 60, 120, 180, 240, 300, 360])
# ax1.set_yticks([-50, 0, 50, 100, 150, 200, 250])
# # plt.savefig('./graph/rnn_prediction.png')
# ax1.legend()
#
#
# ax2.plot(inv_yhat[17:376, 1], label='T2_prediction')
# ax2.plot(inv_y[17:376, 1], label='T2_truth')
# # ax2.set_title("T2", fontsize='16')
# # ax2.set_xlabel("Timestamps", fontsize='14')
# ax2.set_ylabel("PM2.5 (T2)", fontsize='14')
# ax2.set_xticks([0, 60, 120, 180, 240, 300, 360])
# ax2.set_yticks([-50, 0, 50, 100, 150, 200, 250])
# # plt.savefig('./graph/rnn_prediction.png')
# ax2.legend()
#
#
# ax3.plot(inv_yhat[17:376, 2], label='T3_prediction')
# ax3.plot(inv_y[17:376, 2], label='T3_truth')
# # ax3.set_title("T3", fontsize='16')
# ax3.set_xlabel("Timestamps", fontsize='14')
# ax3.set_ylabel("PM2.5 (T3)", fontsize='14')
# ax3.set_xticks([0, 60, 120, 180, 240, 300, 360])
# ax3.set_yticks([-50, 0, 50, 100, 150, 200, 250])
# # plt.savefig('./graph/rnn_prediction.png')
# ax3.legend()
# plt.show()
