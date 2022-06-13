#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import pandas as pd
import pickle
import os
import yahoo_data_tools as ytls
from portfolio_backtest import PortfolioBacktest

# %% Load data

# Load dataset of closing prices
dataset = 'SP500'  # SP500 or STOXXE600
thesis_file = './Thesis/SP500_thesis.pkl'

if os.path.isfile(thesis_file):
    [close_df, return_df, real_vol_df, mktcap_df, rf_ret_df] = pd.read_pickle(thesis_file)
else:
    tsv_file = open("/home/federico/Documents/msfinance/MasterThesis/Matlab/SP500/"
                    "quotes_mids_1min_20180102-20200825.tsv", 'r')
    
    close_df = pd.read_csv(tsv_file, delimiter="\t", header=0)
    
    tsv_file.close()
    
    close_df['Datetime'] = pd.to_datetime(close_df['Datetime'])
    close_df = close_df.set_index(pd.DatetimeIndex(close_df.Datetime))
    close_df = close_df.drop(['Datetime'], axis=1)
    
    date_str_list = ytls.get_date_list(close_df)
    
    ts_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in close_df.index]
    
    close_ds = {}
    for d in date_str_list:
        matching = [d in s for s in ts_str_list]
        close_ds[d] = close_df[matching]
    
    close_df, return_df, real_vol_df = ytls.get_returns_volat(close_ds=close_ds,
                                                              price_df=close_df,
                                                              verbose=True)

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
        pickle.dump([close_df, return_df, real_vol_df, mktcap_df, rf_ret_df], 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    
close_df = close_df[close_df.columns.intersection(mktcap_df.columns)]
return_df = return_df[return_df.columns.intersection(mktcap_df.columns)]
real_vol_df = real_vol_df[real_vol_df.columns.intersection(mktcap_df.columns)]

mkt_idx_df, mkt_ret_df = pd.read_pickle('./SP500/mkt_idx.pkl')

# %% Set parameters and run analysis

endow = 1e6  # Initial amount of money to be invested
n_stk = 30  # Number of stocks to hold in the portfolio
n_obs = 30  # Number of past observations to use as training data
n_reb = 10  # Rate of portfolio rebalancing (in trading days)
algo = 'rmmtm'  # Algorithm to use for stock selection
wght_mtds = ['equal', 'mktcap', 'lotp', 'ilotp']
# wght_mtd = ['riskpar']
trsctn_fee_fix = 0.  # Fixed transaction fees
trsctn_fee_prop = 0e-4  # Proportional transaction fees

pf_backtest = PortfolioBacktest(dataset=dataset, endow=endow, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                algos=algo, wght_mtds=wght_mtds, price_df=close_df,
                                return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                bema_idx_df=mkt_idx_df, risk_free_ret=rf_ret_df, idx_start=0, lag=1,
                                trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                risk_avers_factor=None,
                                multi_proc=True, cv_opt_bw=False,
                                save_stk_hist=False)

pf_backtest.backtest()

pf_backtest.print_results_for_latex()

pf_backtest.plot_cum_wealth()
