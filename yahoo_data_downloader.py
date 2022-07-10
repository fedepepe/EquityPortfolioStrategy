#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
import datetime as dt
import pandas as pd
from pandas_datareader import data as dataread
import pickle
import yfinance as yf
import glob
import yahoo_data_tools as ytls
import os


class YahooDataDownloader:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.data_path = f'./{dataset}'
        self.close_intraday_filename = f'{self.data_path}/close_intraday_adj.pkl'
        self.prices_daily_filename = f'{self.data_path}/prices_daily.pkl'
        self.price_filename_tag = f'{self.data_path}/prices_intraday_'
        self.stk_data_filename = f'{self.data_path}/stk_data.pkl'
        self.stk_data_filename_tag = f'{self.data_path}/stk_data_'
        self.retvol_daily_filename_tag = f'{self.data_path}/retvol_daily_'
        self.mktcap_filename = f'{self.data_path}/mktcap.pkl'
        self.mktcap_filename_tag = f'{self.data_path}/mktcap_'
        self.market_data_filename_tag = f'{self.data_path}/mkt_data_'
        self.ymd_fmt_str = '%Y-%m-%d'
        self.start_date = None
        self.end_date = None

    # %% DOWNLOADING AND STORING DATA SECTION
    def merge_adjust_stk_prices(self):
        # Merge stock data into a single dataset
        close_ds = ytls.merge_stock_prices(self.close_intraday_filename, self.price_filename_tag)

        # Download updated daily closing prices
        date_str_list = list(close_ds.keys())
        date_str_list.sort()
        self.start_date = dt.datetime.strptime(date_str_list[0], self.ymd_fmt_str)
        self.end_date = dt.datetime.strptime(date_str_list[-1], self.ymd_fmt_str)
        self.end_date = self.end_date + dt.timedelta(days=1)
        prices_daily = ytls.download_stock_data(self.dataset, self.start_date, self.end_date, '1d')
        
        # Ensure index is a DatetimeIndex
        prices_daily.index = pd.to_datetime(prices_daily.index.to_series(), utc=True)
        prices_daily = prices_daily.sort_index()
        prices_daily = prices_daily.fillna(method='ffill', limit=3)
        with open(self.prices_daily_filename, 'wb') as handle:
            pickle.dump(prices_daily, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Back-adjust prices to account for dividends and splits
        close_adj_ds = ytls.adjust_stk_prices(close_ds, prices_daily.Close)
        with open(self.close_intraday_filename, 'wb') as handle:
            pickle.dump(close_adj_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def download_stk_prices(self, last_dl_date=None):
        # If not given, find last download date, if any, otherwise set it back to 2 months ago
        if last_dl_date is None:
            if os.path.isfile(self.close_intraday_filename):
                close_adj_ds = pd.read_pickle(self.close_intraday_filename)
                last_dl_date = dt.datetime.strptime(list(close_adj_ds.keys())[-1], self.ymd_fmt_str)
            else:
                last_dl_date = dt.datetime.today() - dt.timedelta(days=60)
        elif type(last_dl_date) is str:
            last_dl_date = dt.datetime.strptime(last_dl_date, self.ymd_fmt_str)
            
        today = dt.datetime.today()
        if last_dl_date >= today - dt.timedelta(hours=12):
            return
            
        data_chunks = []
        end_date = today
        print(' --- Downloading latest stock intraday prices... --- ')        
        while last_dl_date < end_date:
            start_date = end_date - dt.timedelta(days=6)
            if start_date >= today - dt.timedelta(days=30):
                intvl = '1m'
            else:
                intvl = '5m'
            data_chunks.append(ytls.download_stock_data(self.dataset, start_date, end_date, intvl))
            end_date = end_date - dt.timedelta(days=6)
        
        data_chunks.reverse()
        prices_df = pd.concat(data_chunks)
        
        # Ensure index is a DatetimeIndex
        prices_df.index = pd.to_datetime(prices_df.index.to_series(), utc=True)
        prices_df = prices_df.sort_index()
        
        ts = dt.date.today()
        with open(f'{self.price_filename_tag}{ts.strftime(self.ymd_fmt_str)}.pkl', 'wb') as handle:
            pickle.dump(prices_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return prices_df

    def download_mktcap_data(self):       
        print(' --- Downloading market capitalization data...', end=' ')
        ts = dt.date.today()
        [df, ts] = pd.read_pickle(f'{self.stk_data_filename_tag}{ts.strftime(self.ymd_fmt_str)}.pkl')
        if 'marketCap' in df.columns:
            mktcap_df = df['marketCap']
            with open(f'{self.mktcap_filename_tag}{ts.strftime(self.ymd_fmt_str)}.pkl', 'wb') as handle:
                pickle.dump([mktcap_df, ts], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        mktcap_df = ytls.merge_mktcap_data(self.mktcap_filename, self.mktcap_filename_tag)
        with open(self.mktcap_filename, 'wb') as handle:
            pickle.dump(mktcap_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done. --- ')
        
    def download_stk_data(self, tickers):
        print(' --- Downloading full stock data...', end=' ')
        data = pd.DataFrame()
        for tckr in tickers:
            try:
                data = pd.concat([data, dataread.get_quote_yahoo([tckr])])
            except:
                print(f'\nNo quotes available for {tckr}.')
        
        ts = dt.date.today()
        with open(f'{self.stk_data_filename_tag}{ts.strftime(self.ymd_fmt_str)}.pkl', 'wb') as handle:
            pickle.dump([data, ts], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done. --- ')
        
        stk_data_ds = ytls.merge_stk_data(self.stk_data_filename, self.stk_data_filename_tag)
        with open(self.stk_data_filename, 'wb') as handle:
            pickle.dump(stk_data_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def download_latest_data(self, last_dl_date=None):    
        prices_df = self.download_stk_prices(last_dl_date)
        tickers = list(prices_df.Close.columns)
        self.download_stk_data(tickers)        
        # self.download_mktcap_data()
        self.merge_adjust_stk_prices()

        timestamp_string = dt.date.today().strftime(self.ymd_fmt_str)
        mkt_filename = f'{self.market_data_filename_tag}{timestamp_string}.pkl'
        if os.path.isfile(mkt_filename):
            os.remove(mkt_filename)

    # %% LOADING DATA SECTION
    def get_stock_data(self):
        # Load price dataset and compute daily returns and volatility
        close_adj_ds = pd.read_pickle(self.close_intraday_filename)
        close_df = pd.read_pickle(self.prices_daily_filename)['Close']
        
        timestamp_string = dt.date.today().strftime(self.ymd_fmt_str)
        retvol_daily_filename = f'{self.retvol_daily_filename_tag}{timestamp_string}.pkl'

        if os.path.isfile(retvol_daily_filename):
            # Just load data from pickle file
            print(' --- Loading stock data... ', end=' ')
            ret_df, real_vol_df, mktcap_df = pd.read_pickle(retvol_daily_filename)
            print('done. --- ')
        else:
            print(' --- Computing stock data... ', end=' ')
            # Discard days with little or no valid data
            for d in list(close_adj_ds):
                n_valid = np.count_nonzero(~np.isnan(close_adj_ds[d]))
                n_total = np.prod(close_adj_ds[d].shape)
                if n_valid / n_total < 0.4:
                    del close_adj_ds[d]

            ret_df, real_vol_df = ytls.get_returns_volat(close_adj_ds, close_df)

            # Get market capitalization data
            mktcap_df_pkl = pd.read_pickle(self.mktcap_filename)

            # Reindex and interpolate data frame of mkt. cap. data over entire price time range
            mktcap_df = ytls.reindex_by_date(mktcap_df_pkl, ret_df.index, bfill=True)

            with open(retvol_daily_filename, 'wb') as handle:
                pickle.dump([ret_df, real_vol_df, mktcap_df],
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('done. --- ')
        
        # Delete old pickle files (if present)
        old_files = glob.glob(f'{self.retvol_daily_filename_tag}*.pkl')
        for f in old_files:
            if os.path.relpath(f) != os.path.relpath(retvol_daily_filename):
                os.remove(f)

        if self.start_date is None:
            self.start_date = close_df.index[0]
        if self.end_date is None:
            self.end_date = close_df.index[-1] + dt.timedelta(days=1)

        return close_df, ret_df, real_vol_df, mktcap_df, close_adj_ds

    def get_mkt_data(self, date_time_index):

        timestamp_string = dt.date.today().strftime(self.ymd_fmt_str)
        filename = f'{self.market_data_filename_tag}{timestamp_string}.pkl'

        if os.path.isfile(filename):  # Just load data from pickle file
            print(' --- Loading market data... ', end=' ')
            [mkt_ret_df, mkt_idx_df] = pd.read_pickle(filename)
            print('done. --- ')
        else:
            print(' --- Downloading market data... ', end=' ')
            # Set start and end date for download
            if date_time_index is None:
                start_date = self.start_date
                end_date = self.end_date
            else:
                start_date = date_time_index[0] - dt.timedelta(days=3)
                end_date = date_time_index[-1] + dt.timedelta(days=1)

            # Set asset symbol for download
            if self.dataset == 'STOXXE600':
                symbol = '^STOXX'
            elif self.dataset == 'SP500':
                symbol = '^SP500TR'
            else:
                return None

            # Download index data
            data = yf.download(symbol, start=start_date, end=end_date, interval='1d',
                               auto_adjust=False, actions=False, threads=False)

            # Index value and returns
            mkt_idx_df_tmp = data['Close']
            mkt_ret_df_tmp = mkt_idx_df_tmp.pct_change(fill_method=None)

            # Reindex the two series
            mkt_idx_df = ytls.reindex_by_date(mkt_idx_df_tmp, date_time_index)
            mkt_ret_df = ytls.reindex_by_date(mkt_ret_df_tmp, date_time_index)

            # Replace missing values with zeros for returns and last values for index
            mkt_ret_df = mkt_ret_df.fillna(value=0)
            mkt_idx_df = mkt_idx_df.fillna(method='ffill')

            with open(filename, 'wb') as handle:
                pickle.dump([mkt_ret_df, mkt_idx_df], handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('done. --- ')
            
        # Delete old pickle files
        old_files = glob.glob(f'{self.market_data_filename_tag}*.pkl')
        for f in old_files:
            if f != filename:
                os.remove(f)

        return mkt_ret_df, mkt_idx_df
