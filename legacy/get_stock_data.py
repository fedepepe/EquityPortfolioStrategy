#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 18:15:23 2021

@author: federico
"""

import pandas as pd
from yahoo_data_downloader import YahooDataDownloader
from pandas_datareader import data as dataread
import datetime as dt
import pickle


# Load dataset of closing prices
dataset = 'SP500'  # SP500 or STOXXE600
data_path = './' + dataset + '/'

YahooDataDownloaderObj = YahooDataDownloader(dataset)

last_price_df, return_df, real_vol_df, mktcap_df, close_adj_ds = YahooDataDownloaderObj.get_stock_data()

data = pd.DataFrame()
for tckr in list(last_price_df.columns):
    try:
        data = data.append(dataread.get_quote_yahoo([tckr]))
    except:
        print(tckr + ' not found.')

ts = dt.date.today()
ymd_format_str = '%Y-%m-%d'
with open(data_path + f'data_{ts.strftime(ymd_format_str)}.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
