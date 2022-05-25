#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:28:46 2021

@author: federico
"""

import pandas as pd
import glob
import pickle

dataset = 'SP500' # SP500 or STOXXE600
csv_path = '/media/windata/YahooFinanceData/' + dataset + '/OHLCV/'
out_path = './' + dataset + '/'

csv_file_collection = glob.glob(csv_path + dataset + '*.csv')
csv_file_collection.sort()

for csv_file in csv_file_collection:
    with open(csv_file, newline='') as file_handle:
        # results_reader = csv.reader(results_handle, delimiter=',')
        results_df = pd.read_csv(file_handle, header=[0,1], index_col=0)

    date_str = csv_file[-14:-10] + '-' + csv_file[-9:-7] + '-' + csv_file[-6:-4]
        
    df_datetime_idx = pd.to_datetime(results_df.index, utc=False)
    results_df.set_index(pd.DatetimeIndex(df_datetime_idx), inplace=True)
            
    with open(out_path + f'data_{date_str}.pkl', 'wb') as handle:
        pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # close_df = results_df['Close']