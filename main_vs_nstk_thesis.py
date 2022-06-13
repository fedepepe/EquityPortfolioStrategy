#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
import pandas as pd
from yahoo_data_downloader import YahooDataDownloader
from portfolio_backtest import PortfolioBacktest
import portfolio_plot as pf_plot
import pickle
import os
import time

# %% Load data

# Load dataset of closing prices
dataset = 'SP500'  # SP500 or STOXXE600

YahooDataDownloaderObj = YahooDataDownloader(dataset)

if os.path.isfile('SP500_thesis.pkl'):
    [last_price_df, return_df, real_vol_df, mktcap_df, rf_ret_df] = pd.read_pickle('SP500_thesis.pkl')
else:
    tsv_file = open("/home/federico/Documents/msfinance/MasterThesis/Matlab/SP500/quotes_mids_1min_20180102-20200825.tsv", 'r')
    
    close_df = pd.read_csv(tsv_file, delimiter="\t", header=0)
    
    tsv_file.close()
    
    close_df['Datetime'] = pd.to_datetime(close_df['Datetime'])
    close_df = close_df.set_index(pd.DatetimeIndex(close_df.Datetime))
    close_df = close_df.drop(['Datetime'], axis=1)
    
    date_str_list = YahooDataDownloaderObj.get_date_list(close_df)
    
    ts_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in close_df.index]
    
    close_ds = {}
    for d in date_str_list:
        matching = [d in s for s in ts_str_list]
        close_ds[d] = close_df[matching]
    
    last_price_df, return_df, real_vol_df = YahooDataDownloaderObj.get_returns_volat(close_ds, verbose=True)
    
    mktcap_file = open("/home/federico/Documents/msfinance/MasterThesis/Matlab/SP500/MktCapWghts_intp.csv", 'r')
    mktcap_df = pd.read_csv(mktcap_file, header=0)
    mktcap_file.close()
    
    mktcap_df['Date'] = pd.to_datetime(mktcap_df['Date'], utc=False)
    mktcap_df = mktcap_df.set_index(pd.DatetimeIndex(mktcap_df.Date))
    mktcap_df = mktcap_df.drop(['Date'], axis=1)
    
    mktcap_df = mktcap_df.reindex(index=return_df.index, method='nearest')
    mktcap_df = mktcap_df.interpolate(axis=0, method='time')
    
    rf_ret_file = open("/home/federico/Documents/msfinance/MasterThesis/Matlab/SP500/RfRetns.csv", 'r')
    rf_ret_df = pd.read_csv(rf_ret_file, header=0)
    rf_ret_file.close()

    rf_ret_df['Date'] = pd.to_datetime(rf_ret_df['Date'], utc=False)
    rf_ret_df = rf_ret_df.set_index(pd.DatetimeIndex(rf_ret_df.Date))
    rf_ret_df = rf_ret_df.drop(['Date'], axis=1)
    
    rf_ret_df = rf_ret_df.reindex(index=return_df.index, method='nearest')
    rf_ret_df = rf_ret_df.interpolate(axis=0, method='time')
    rf_ret_df = pow(1. + rf_ret_df['Rate'] / 100, 1/252) - 1
    
    with open('SP500_thesis.pkl', 'wb') as handle:
        pickle.dump([last_price_df, return_df, real_vol_df, mktcap_df, rf_ret_df], 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    
last_price_df = last_price_df[last_price_df.columns.intersection(mktcap_df.columns)]
return_df = return_df[return_df.columns.intersection(mktcap_df.columns)]
real_vol_df = real_vol_df[real_vol_df.columns.intersection(mktcap_df.columns)]

mkt_idx_df, mkt_ret_df = pd.read_pickle('./SP500/mkt_idx.pkl')

# %% Set parameters and initialize
endow = 1e6  # Initial amount of money to be invested
n_stk_ar = np.arange(5, 105, 5)  # Number of stocks to hold in the portfolio
n_obs = 45  # Number of past observations to use as training data
n_reb = 15  # Rate of portfolio rebalancing (in trading days)
algos = ['rmsev', 'sev', 'mtm', 'rmmtm']    # Algorithm(s) to use for stock selection
wght_mtds = ['equal', 'mktcap', 'lotp', 'ilotp'] # Weighting method for stock alloc.
# wght_mtd = ['equal', 'riskpar']
trsctn_fee_fix = 0.  # Fixed transaction fees
trsctn_fee_prop = 0.  # Proportional transaction fees

for algo in algos:
    results_dir = './' + dataset + '/results/'
    results_tag = 'thesis_sw_'
    
    pf_backtest = PortfolioBacktest(n_stk=n_stk_ar[0], n_obs=n_obs, n_reb=n_reb,
                                    algo=algo, wght_mtds=wght_mtds, last_price_df=last_price_df,
                                    return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                    risk_free_ret=rf_ret_df, endow=endow, idx_start=100, lag=1,
                                    trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                    risk_avers_factor=None,
                                    multi_proc=True, cv_opt_bw=False,
                                    results_dir=results_dir, results_tag=results_tag)
    
    #%% Simulate
    start = time.time()
    
    # Main loop
    n_sim = len(n_stk_ar)
    n_done = 0
    for n_stk in n_stk_ar:
        pf_backtest.n_stk = n_stk
        
        print(f'\n --- Running sim. {n_done+1} of {n_sim} ---')
    
        pf_backtest.backtest_single()
        
        end_curr = time.time()
        elapsed = end_curr - start  # Total time elapsed
    
        # Analyze portfolio performance
        pf_backtest.analyze(mkt_ret_df=mkt_ret_df, risk_free_ret=rf_ret_df)
    
        # Print results to text file
        pf_backtest.print_results(to_file=True)
    
        n_done += 1
        elapsed_mean = elapsed / n_done
        
        print('\n --- Total time elapsed: ' + time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed)) + \
              ' (average sim. time: ' + time.strftime('%Mm %Ss', time.gmtime(elapsed_mean)) + ') --- \n')
    
    # %% Print market portfolio performance
    # pf_backtest.analyze_print_mkt_results(bema_idx_df, bema_ret_df)

