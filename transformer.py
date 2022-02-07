#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: transformer.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/13/21 16:29
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
from matplotlib import pyplot

from pandas import read_csv

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)

input_window = 96  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=8, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            # print('a',src.size())
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        # print('j',src.size(),self.src_mask.size())
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask
        output = self.decoder(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, input_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - input_window):
        train_seq = input_data[i: i+input_window]
        train_label = input_data[i+output_window: i+input_window+output_window]
        # train_seq = np.append(input_data[i:i+input_window, :][:-output_window, :], np.zeros((output_window, 8)), axis=0)
        # train_label = input_data[i:i+input_window, :]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data():
    # construct a littel toy dataset
    # time = np.arange(0, 400, 0.1)
    # amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # series = read_csv('data/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    data = read_csv('data/pollution1.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    data = data.iloc[:, 0]
    values = data.values

    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    dataset = pd.DataFrame(values)
    data_values = dataset.to_numpy()
    norm_data = scaler.fit_transform(data_values)

    # norm_data = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    # amplitude = scaler.fit_transform(amplitude.to_numpy.reshape(-1, 1)).reshape(-1)

    split = int(len(values)*0.7)
    train_data = norm_data[:split]      # rows * 8
    test_data = norm_data[split:]       # row * 8

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ''lr {:02.6f} | {:5.2f} ms | ''loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 256
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source)-1, eval_batch_size):
            data, target = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += criterion(output, target.reshape(-1, 1).reshape(-1)).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    len(test_result)
    test_result_ = scaler.inverse_transform(test_result[:800])
    truth_ = scaler.inverse_transform(truth[:800])
    print(test_result.shape, truth.shape)

    pyplot.plot(test_result_)
    pyplot.plot(truth_)
    # pyplot.plot(test_result - truth, color="green")
    # pyplot.grid(True, which='both')
    pyplot.title('Prediction V.S. Actual Value')
    pyplot.axhline(y=0, color='k')
    pyplot.legend()
    # pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.show()
    pyplot.close()

    return total_loss / i


# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    # visualize if the model picks up any long therm structure within the data.
    pyplot.plot(data[:input_window], color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.savefig('graph/transformer-future%d.png' % steps)
    # pyplot.show()
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source)-1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

# get dataset
train_data, val_data, scaler = get_data()

# get model
model = TransAm().to(device)

# define loss, optimizer
criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# train model
best_val_loss = float("inf")
epochs = 10  # The number of epochs
best_model = None

all_train_loss, all_val_loss = [], []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    all_train_loss.append(evaluate(model, train_data))
    all_val_loss.append(evaluate(model, val_data))
    # train_loss = evaluate(model, train_data)
    # val_loss = evaluate(model, val_data)

    if (epoch % 5 is 0):
        # train_loss = plot_and_loss(model, train_data, epoch)
        val_loss = plot_and_loss(model, val_data, epoch, scaler)
        # predict_future(model, val_data, 200)
    else:
        train_loss = evaluate(model, train_data)
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), train_loss, math.exp(train_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()



pyplot.plot(all_train_loss)
pyplot.plot(all_val_loss)
pyplot.legend()
pyplot.show()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length, batch size, feature number)
# out = model(src)
#
# print(out)
# print(out.shape)