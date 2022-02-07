#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: baseline.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/25/22 11:48
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from helper import series_to_supervised, mean_absolute_percentage_error


# load dataset
dataset = pd.read_csv('data/pollution1.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])

# ensure all data is float
values = values.astype('float32')

# plot Prediction V.S. actual value of PM2.5
# 30688 in codes --> first test data --> 30690 in excel
# 30688 in codes --> 0 --> 30690 in excel
# 30705 in codes --> 17 --> 30707 in excel --> 2013-07-04-09:00
# 31064 in codes --> 376 --> 31066 in excel --> 2013-07-19-8:00