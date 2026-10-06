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
import time
import pickle

# % Load data

# Load dataset of closing prices
for dataset in ['SP500', 'STOXXE600']:

    YahooDataDownloaderObj = YahooDataDownloader(dataset)
    
    # Download latest stock prices
    YahooDataDownloaderObj.download_latest_data()
    # YahooDataDownloaderObj.merge_adjust_stk_prices()
    
    close_df, return_df, real_vol_df, mktcap_df, close_adj_ds = YahooDataDownloaderObj.get_stock_data()
    
    mkt_ret_df, mkt_idx_df = YahooDataDownloaderObj.get_mkt_data(return_df.index)