# import pandas as pd
# import numpy as np
# from matplotlib import pyplot
# import matplotlib.pyplot as plt
#
#
#
# dataset = pd.read_excel('data/transformer_record.xlsx')
# # print(dataset)
# data = dataset.loc[:, ['truth', 't1', 't2', 't3']]
# print(data)
#
# dates = ['07-04', '07-07', '07-10', '07-13', '07-16', '07-19']
#
# # pyplot.rcParams['font.family'] = 'serif'
# # pyplot.rcParams['font.serif'] = ['Times New Roman'] + pyplot.rcParams['font.serif']
# pyplot.plot(data.iloc[:, 1], label='T1_prediction')
# pyplot.plot(data.iloc[:, 2], label='T2_prediction')
# pyplot.plot(data.iloc[:, 3], label='T3_prediction')
# pyplot.plot(data.iloc[:, 0], label='ground_truth')
#
# pyplot.title("Predicted (T1, T2, T3) vs. Actual Value of PM2.5", fontsize='16')
# pyplot.xlabel('Time', fontsize='14')
# pyplot.ylabel('PM2.5', fontsize='14')
# pyplot.legend(prop={"size": 12}, loc='upper right')
# pyplot.xticks(np.arange(0, 361, 72), dates, fontsize=14)
# pyplot.yticks(fontsize=14)
# # pyplot.savefig('graph/multi-transformer.png', dpi=300)
# pyplot.show()
# pyplot.close()
#
#
#
# # show the difference
# plt.plot(np.subtract(data.iloc[:, 1], data.iloc[:, 0]), label='T1_difference')     # truth_y[19:378]
# plt.plot(np.subtract(data.iloc[:, 2], data.iloc[:, 0]), label='T2_difference')
# plt.plot(np.subtract(data.iloc[:, 3], data.iloc[:, 0]), label='T3_difference')
#
#
# plt.plot(dataset.loc[:, 't1error'], label='T1_difference')
# plt.plot(dataset.loc[:, 't2error'], label='T2_difference')
# plt.plot(dataset.loc[:, 't3error'], label='T3_difference')
# plt.title("Difference between Predicted and Actual Value of PM2.5", fontsize=16)
# plt.xlabel("Time", fontsize='14')
# plt.ylabel("Difference", fontsize='14')
# plt.xticks(np.arange(0, 361, 72), dates, fontsize=14)
# plt.yticks(fontsize=14)
# # plt.savefig('./graph/RNN_prediction.png')
# plt.legend(prop={"size": 12}, loc='upper right')
# plt.show()
# plt.close()


from math import sqrt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.layers.core import Dense, Dropout
from helper import series_to_supervised, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam
import numpy as np

# load dataset
dataset = read_csv('data/pollution1.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# specify the number of lag hours
n_hours = 2*24
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
lr = 0.0005
EPOCHS = 100
# design network
model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.summary()
model.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
              # optimizer='adam',
              loss='mse',
              metrics=['mae'])

# fit network
history = model.fit(train_X, train_y,
                    epochs=EPOCHS,
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

# invert scaling for forecast
inv_yhat = scaler.inverse_transform(yhat)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = scaler.inverse_transform(test_y)
print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)

# test_y.to_csv('./doc/singlestep/test_y.csv')
# inv_yhat.to_csv('./doc/singlestep/inv_yhat_lstm.csv')

# calculate RMSE, MAE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
mape = mean_absolute_percentage_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
print('Test MAPE: %.3f' % mape)


# plot Prediction V.S. actual value of PM2.5
# [] <-> 30688 in codes --> first test data --> 30690 in excel
# [] <-> 30688 in codes --> 0 --> 30690 in excel
# [] <-> 30705 in codes --> 17 --> 30707 in excel --> 2013-07-04-09:00
# [] <-> 31064 in codes --> 376 --> 31066 in excel --> 2013-07-19-8:00

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


dates = ['07-04', '07-07', '07-10', '07-13', '07-16', '07-19']
plt.plot(inv_yhat[10:369], label='prediction')
plt.plot(inv_y[10:369], label='ground_truth')
plt.title("Predicted vs. Actual Value of PM2.5", fontsize='16')
plt.xlabel('Time', fontsize='14')
plt.ylabel('PM2.5', fontsize='14')
plt.legend(prop={"size":14}, loc='upper right')
plt.xticks(np.arange(0, 361, 72), dates, fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('graph/rnn_prediction.png', dpi=300)
plt.show()