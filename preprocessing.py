#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: preprocessing.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 12/15/21 12:44
"""


from pandas import read_csv
from datetime import datetime

# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('data/raw_uci.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:]


# save to file
dataset.to_csv('./data/pollution1.csv')

print(dataset.head(5))
print('Done')