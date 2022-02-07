#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : TSF
@ FileName: plot_feature_trend.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/24/22 21:51
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

dataset = pd.read_csv('data/pollution1.csv')
print(dataset)
dataset['time'] = pd.to_datetime(dataset['date']).dt.date
data = dataset[:]
print(data)

fig, axs = plt.subplots(7, 1, figsize=(18, 16), constrained_layout=True)

features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']

i = 0
for ax in axs:
    ax.plot('time', features[i], data=data)
    ax.grid(axis="x")
    ax.set_title(features[i], loc='left', y=0.65, x=0.01, fontsize=28)
    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.tick_params(axis='x', which='minor', labelsize=26)
    # ax.tick_params(axis='y', colors='black')  # setting up Y-axis tick color to black
    ax.tick_params(axis='x', colors='black')  # setting up X-axis tick color to red
    # ax.set_xlabel('Time')

    # for label in ax.get_xticklabels(which='major'):
    #     label.set(rotation=30)
    i += 1

plt.xlabel('Time', fontsize=35)
# plt.savefig('graph/Figure7.png', dpi=300)   # if want to save, then need to comment plt.show()

    
plt.show()