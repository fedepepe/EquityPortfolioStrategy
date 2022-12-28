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

# %% Set parameters and run analysis

endow = 1e6  # Initial amount of money to be invested
# n_stk = 30  # Number of stocks to hold in the portfolio
n_obs = 30  # Number of past observations to use as training data
n_reb = 10  # Rate of portfolio rebalancing (in trading days)
algo = 'rmsev'  # Algorithm to use for stock selection
wght_mtds = ['mktcap'] # ['equal', 'mktcap', 'lotp', 'ilotp']
# wght_mtd = ['riskpar']
trsctn_fee_fix = 0.  # Fixed transaction fees
trsctn_fee_prop = 0e-4  # Proportional transaction fees

n_factors = [0, 1, 3, 4]
L = ['& Metric        ', '& Excess return ', '&               ', 
     '& CAPM alpha    ', '&               ', '& 3F alpha      ', '&               ',
     '& 4F alpha      ', '&               ']

for n_stk in range(1, 7):
    pf_backtest = PortfolioBacktest(endow=endow, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                    algo=algo, wght_mtds=wght_mtds, last_price_df=last_price_df,
                                    return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                    risk_free_ret=rf_ret_df, idx_start=0, lag=1,
                                    trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                    risk_avers_factor=None,
                                    multi_proc=True, cv_opt_bw=False,
                                    save_stk_hist=False)
    
    pf_backtest.backtest_single()
    
    # Analyze portfolio performance
    pf_backtest.analyze(mkt_ret_df=mkt_ret_df)
    
    pf_backtest.print_results_for_latex()
    
    pf_dict = pf_backtest.portfolios
    
    # %% Fama-French factors
    for n in range(len(n_factors)):
        pf_backtest.compute_ff_factors(n_factors[n])
        
        wm = wght_mtds[0]
        
        L[2 * n + 1] += f" & {pf_dict[wm].ff_factors['constant']['coeff']:.2f}"
        pval_str = f" & ({pf_dict[wm].ff_factors['constant']['t_value']:.2f})"
        if pf_dict[wm].ff_factors['constant']['p_value'] <= 0.01:
            pval_str += '***'
        elif pf_dict[wm].ff_factors['constant']['p_value'] <= 0.05:
            pval_str += '**'
        elif pf_dict[wm].ff_factors['constant']['p_value'] <= 0.1:
            pval_str += '*'
        pval_str += f" ({pf_dict[wm].ff_factors['constant']['p_value']:.5f})"
        
        L[2 * n + 2] += pval_str
        if n_stk == 6:
            L[2 * n + 1] += ' \\\\\n'
            L[2 * n + 2] += ' \\\\\n'
            
    with open(f'./{algo}_q{n_stk}_metric.txt') as f:
        array = [float(line) for line in f]
        L[0] += f' & {sum(array)/len(array):.5f}'
    if n_stk == 6:
        L[0] += ' \\\\\n'
        
# %% Print Latex file
filename = f'./{algo}_{wm}_stat.txt'
file_handle = open(filename, 'w')
file_handle.writelines(L)
file_handle.close()

# %% Plot
fig_wealth = None
for wm in wght_mtds:
    fig_wealth = pf_plot.plot_cum_wealth(pf_dict[wm].val_tot_hist, fig_wealth, descr='wght=' + wm)
    
fig_wealth = pf_plot.plot_mkt_index(pf_dict[wght_mtds[0]].val_tot_hist, mkt_idx_df, fig_wealth)

fig_wealth.axes[0].legend(loc='best')
fig_wealth.show()
