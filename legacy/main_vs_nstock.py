#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
from yahoo_data_downloader import YahooDataDownloader
from portfolio_sim import PortfolioSimulator
import portfolio_plot as pf_plot
import time

# %% Load data

# Load dataset of closing prices
dataset = 'STOXXE600'  # SP500 or STOXXE600

YahooDataDownloaderObj = YahooDataDownloader(dataset)

# Download latest stock prices
YahooDataDownloaderObj.download_latest_data()

last_price_df, return_df, real_vol_df, mktcap_df, close_adj_ds = YahooDataDownloaderObj.get_stock_data()

mkt_ret_df, mkt_idx_df = YahooDataDownloaderObj.get_mkt_data(return_df.index)

# %% Set parameters and initialize

endow = 1e6  # Initial amount of money to be invested
n_stk_ar = np.arange(1, 31, 1)  # Number of stocks to hold in the portfolio
n_obs = 11  # Number of past observations to use as training data
n_reb = 18  # Rate of portfolio rebalancing (in trading days)
algo = 'rmsev'  # Algorithm to use for stock selection
# Weighting method(s) for allocation
wght_mtd = ['equal', 'metric', 'mktcap', 'riskpar', 'lotp', 'tp']
trsctn_fee_fix = 2  # Fixed transaction fees
trsctn_fee_prop = 3e-4  # Proportional transaction fees

results_dir, results_tag = './' + dataset + '/results/', 'sw'

PfSimObj = PortfolioSimulator(n_stk=n_stk_ar[0], n_obs=n_obs, n_reb=n_reb,
                              algo=algo, wght_mtd=wght_mtd, last_price_df=last_price_df,
                              return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                              endow=endow, trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                              risk_avers_factor=np.inf,
                              multi_proc=True, cv_opt_bw=False, results_dir=results_dir,
                              results_tag=results_tag)

#%% Simulate
start = time.time()

# Main loop
n_sim = len(n_stk_ar)
n_done = 0
for n_stk in n_stk_ar:
    PfSimObj.n_stk = n_stk

    start_curr = time.time()
    # Run simulation
    PfSimObj.backtest_single()
    end_curr = time.time()

    elapsed_curr = end_curr - start_curr  # Time spent in current simulation run
    elapsed_tot = end_curr - start  # Total time elapsed

    # Analyze portfolio performance
    PfSimObj.analyze(mkt_ret_df)

    # Print results to text file
    PfSimObj.print_results()

    n_done += 1
    elapsed_mean = elapsed_tot / n_done

    # print(f'\n --- Done sim. {n_done} of {n_sim} in ' +
    #       time.strftime('%Mm %Ss', time.gmtime(elapsed_curr)) + ' --- ')
    print('\n --- Total time elapsed: ' + time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed_tot)) + \
          ' (average sim. time: ' + time.strftime('%Mm %Ss', time.gmtime(elapsed_mean)) + ') --- \n')

# %% Plot
figure = None
for wm in wght_mtd:
    results_df = PfSimObj.load_results(wght_mtd=wm, filename=None)
    figure = pf_plot.plot_vs_nstock(results_df=results_df, label='alpha', figure=figure, descr=wm)
    
figure.axes[0].legend(loc='best')
figure.tight_layout()
figure.show()
