#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: draft.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/13/21 17:32
"""

# importing required modules
import matplotlib.pyplot as plt

# creating plotting data
xaxis = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
yaxis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# plotting
plt.plot(xaxis, yaxis)
plt.xlabel("X")
plt.ylabel("Y")

# saving the file.Make sure you
# use savefig() before show().
plt.savefig("squares.png")

plt.show()

# # load data
# def parse(x):
#     return datetime.strptime(x, '%Y %m %d %H')
#
# dataset = read_csv('data/raw_uci.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
#
# dataset.drop('No', axis=1, inplace=True)
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# dataset['pollution'].fillna(0, inplace=True)
# dataset = dataset[24:]
#
#
# # save to file
# dataset.to_csv('./data/pollution1.csv')
#
# print(dataset.head(5))
# print('Done')

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     """
#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations as a list or NumPy array.
#         n_in: Number of lag observations as input (X).
#         n_out: Number of observations as output (y).
#         dropnan: Boolean whether or not to drop rows with NaN values.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning.
#     """
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
#
# raw = DataFrame()
# raw['ob1'] = [x for x in range(10)]
# raw['ob2'] = [x for x in range(50, 60)]
# values = raw.values
# data = series_to_supervised(values, 3, 2)
# print(data)
