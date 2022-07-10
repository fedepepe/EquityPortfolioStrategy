#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:51:00 2021

@author: federico
"""

import numpy as np
import pandas as pd
import symbols_string
import yfinance as yf
import os
import glob
import statistics
from typing import Union
# import tabula

ymd_date_fmt = '%Y-%m-%d'


def print_status_msg(msg):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f' --- {msg} ... ', end=' ')
            result = function(*args, **kwargs)
            print(' done. --- ')
            return result
        return wrapper
    return decorator


def get_date_list(close_df: pd.DataFrame) -> list:
    date_str_list = close_df.index
    date_str_list = [d.strftime(ymd_date_fmt) for d in date_str_list]
    date_str_list = list(set(date_str_list))
    date_str_list.sort()
    return date_str_list


def ts_time_rounder(ts):
    # Rounds time of datetime to midnight
    return ts.replace(second=0, microsecond=0, minute=0, hour=0)


def get_tickers(dataset: str) -> list:
    if dataset == 'STOXXE600':
        # Get string with tickers stored in a file
        symbols_str = symbols_string.get_symbols_string_STOXXE600().split()
        return symbols_str

        # stoxxe600_file = 'https://www.stoxx.com/document/Reports/SelectionList/2022/January/sl_sx5e_202201.pdf'
        # stoxxe600_comp = tabula.read_pdf(stoxxe600_file, pages='all', multiple_tables=False)[0]
        # symbols_pdf = stoxxe600_comp.RIC.tolist()
        # return symbols_str + list(set(symbols_pdf) - set(symbols_str))

        # or, altenatively, get updated tickers from Dividendmax
        # page = 1
        # last_page = False
        # se600_comp_df = pd.DataFrame()
        # while not last_page:
        #     se600_url = 'https://www.dividendmax.com/market-index-constituents/stoxx600?page='+ str(page)
        #     se_comp = pd.read_html(se600_url, header=0)[0]
        #     if se_comp.shape[0] > 1:
        #         se600_comp_df = se600_comp_df.append(se_comp)
        #         page += 1
        #     else:
        #         last_page = True
        # se600_comp_df.index = np.arange(0, len(se600_comp_df))

        # yahoo_suffix_dict = {'Athens Exchange':             'AT',
        #                      'Barcelona Stock Exchange':    'MC',
        #                      'Berlin Stock Exchange':       'DE',
        #                      'Copenhagen Stock Exchange':   'CO',
        #                      'Euronext Amsterdam':          'AS',
        #                      'Euronext Brussels':           'BR',
        #                      'Euronext Lisbon':             'LS',
        #                      'Euronext Paris':              'PA',
        #                      'Frankfurt Stock Exchange':    'DE',
        #                      'Helsinki Stock Exchange':     'HE',
        #                      'Irish Stock Exchange':        'IR',
        #                      'Italian Stock Exchange':      'MI',
        #                      'Johannesburg Stock Exchange': 'JO',
        #                      'London Stock Exchange':       'L',
        #                      'Luxembourg Stock Exchange':   'DE',
        #                      'Madrid Stock Exchange':       'MC',
        #                      'New York Stock Exchange':     '',
        #                      'Oslo Stock Exchange':         'OL',
        #                      'SIX Swiss Exchange':          'SW',
        #                      'Stockholm Stock Exchange':    'ST',
        #                      'Valencia Stock Exchange':     'MC',
        #                      'Vienna Stock Exchange':       'VI',
        #                      'Warsaw Stock Exchange':       'WA',
        #                      'Xetra':                       'DE'}

        # for i in se600_comp_df.index:
        #     se600_comp_df.loc[i, 'Ticker'] = se600_comp_df.loc[i, 'Ticker'].strip('.')
        #     if yahoo_suffix_dict[se600_comp_df.loc[i, 'Exchange']]:
        #         se600_comp_df.loc[i, 'Ticker'] = se600_comp_df.loc[i, 'Ticker'] + '.' + \
        #             yahoo_suffix_dict[se600_comp_df.loc[i, 'Exchange']]
        #     se600_comp_df.loc[i, 'Ticker'] = se600_comp_df.loc[i, 'Ticker'].replace(' ', '-')

        # symbols = se600_comp_df['Ticker'].to_list()

    elif dataset == 'SP500':
        # Get string with tickers stored in a file
        # symbols = symbols_string.get_symbols_string_SP500()

        # or, alternatively, get updated tickers from Wikipedia
        sp500_wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_constituents = pd.read_html(sp500_wiki_url, header=0)[0]
        return sp500_constituents.Symbol.tolist()
    else:
        return []


@print_status_msg('Downloading data from Yahoo Finance')
def download_stock_data(dataset, date_start, date_end, intrvl_str):
    symbols = get_tickers(dataset)
    df = yf.download(symbols, start=date_start, end=date_end, interval=intrvl_str,
                     threads=True, auto_adjust=True, actions=True)
    df = df.dropna(how='all')
    return df


def reindex_by_date(df_old: Union[pd.Series, pd.DataFrame],
                    idx_new: pd.Index,
                    bfill: bool = False):
    if isinstance(df_old, pd.Series):
        df_new = pd.Series(index=idx_new, dtype=float)
        for ts in idx_new:
            matching = [y and m and d for y, m, d in zip(df_old.index.year == ts.year,
                                                         df_old.index.month == ts.month,
                                                         df_old.index.day == ts.day)]
            if any(matching):
                df_new[ts] = df_old[matching].values

    elif isinstance(df_old, pd.DataFrame):
        df_new = pd.DataFrame(index=idx_new, columns=df_old.columns)
        for ts in idx_new:
            matching = [y and m and d for y, m, d in zip(df_old.index.year == ts.year,
                                                         df_old.index.month == ts.month,
                                                         df_old.index.day == ts.day)]
            if any(matching):
                df_new.loc[ts, :] = df_old.loc[matching, :].values

        df_new[df_new.columns] = df_new[df_new.columns].apply(pd.to_numeric)
        df_new = df_new.interpolate(method='time')
        if bfill:
            df_new = df_new.fillna(method='bfill')

    return df_new


@print_status_msg('Building price dataset')
def merge_stock_prices(close_adj_filename, price_filename_tag):
    if os.path.isfile(close_adj_filename):
        close_ds = pd.read_pickle(close_adj_filename)
    else:
        close_ds = {}

    data_file_collection = glob.glob(f'{price_filename_tag}*.pkl')
    data_file_collection.sort(reverse=True)

    for data_file in data_file_collection:
        close_df = pd.read_pickle(data_file).Close

        # Drop spurious rows with almost all nans
        close_df = close_df.dropna(axis=0, thresh=int(.1 * len(close_df.columns)))

        # Replace inf with nan
        close_df = close_df.replace([np.inf, -np.inf], np.nan)

        date_str_list = get_date_list(close_df)

        ts_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in close_df.index]

        for d in date_str_list:
            matching = [d in s for s in ts_str_list]
            close_df_curr = close_df[matching]

            if d in close_ds:  # data frame for day d is already in dataset
                # update current data frame only if the new one carries more data
                n_data_close_df_curr = np.count_nonzero(~np.isnan(close_df_curr))
                n_data_close_ds_d = np.count_nonzero(~np.isnan(close_ds[d]))
                if n_data_close_df_curr > n_data_close_ds_d:
                    close_ds[d] = close_df_curr
            else:
                close_ds[d] = close_df_curr

    return close_ds


@print_status_msg('Back-adjusting stock prices. This may take a while')
def adjust_stk_prices(close_ds, close_daily_df):
    date_str_list = list(close_ds.keys())
    date_str_list.sort()

    # Fill some nan values by interpolating
    close_daily_df_intp = close_daily_df.interpolate(limit=1)

    for d in date_str_list:
        df_curr = close_ds[d]
        tckr_list = df_curr.columns
        ts_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in close_daily_df_intp.index]
        matching = [d in s for s in ts_str_list]
        d_ts = close_daily_df_intp.index[matching]
        for tckr in tckr_list:
            # Check if we have the time series associated to tckr
            if tckr in close_daily_df_intp.columns:
                try:
                    close_daily_curr = float(close_daily_df_intp.loc[d_ts, tckr])

                    # Check if we have a valid adjusted price
                    if not np.isnan(close_daily_curr):
                        # If yes, compute the scale factor and scale
                        price_ser_curr = df_curr[tckr]
                        last_idx = price_ser_curr.last_valid_index()

                        # Check if we have a valid time series to be adjusted
                        if isinstance(last_idx, pd.Timestamp):
                            last_valid_price = price_ser_curr.loc[last_idx]
                            adj_factor = close_daily_curr / last_valid_price
                            price_ser_adj = adj_factor * price_ser_curr
                            df_curr[tckr] = price_ser_adj[~price_ser_adj.index.duplicated(keep='first')]

                except:
                    breakpoint()
                    raise Exception('Problems with data from Yahoo Finance. Try again later.')

        close_ds[d] = df_curr

    return close_ds


@print_status_msg('Building stock data dataset')
def merge_stk_data(stk_data_filename, stk_data_filename_tag):
    if os.path.isfile(stk_data_filename):
        stk_data_ds = pd.read_pickle(stk_data_filename)
    else:
        stk_data_ds = {}

    data_file_collection = glob.glob(f'{stk_data_filename_tag}*.pkl')
    data_file_collection.sort(reverse=True)

    for data_file in data_file_collection:
        [df, ts] = pd.read_pickle(data_file)
        date_str = ts.strftime(ymd_date_fmt)
        stk_data_ds[date_str] = df
    return stk_data_ds


def merge_mktcap_data(mktcap_filename, mktcap_filename_tag):
    if os.path.isfile(mktcap_filename):
        mktcap_df = pd.read_pickle(mktcap_filename)
    else:
        mktcap_df = pd.DataFrame()

    data_file_collection = glob.glob(f'{mktcap_filename_tag}*.pkl')
    data_file_collection.sort()

    for data_file in data_file_collection:
        [df, ts] = pd.read_pickle(data_file)
        if pd.to_datetime(ts, format=ymd_date_fmt) not in mktcap_df.index:
            df.name = ts
            mktcap_df = pd.concat([mktcap_df, df])

    df_datetime_idx = pd.to_datetime(mktcap_df.index.to_series(), format=ymd_date_fmt)

    mktcap_df.set_index(pd.DatetimeIndex(df_datetime_idx), inplace=True)
    return mktcap_df


def get_returns_volat(close_ds, price_df, verbose=False):
    date_str_list = list(close_ds.keys())
    date_str_list.sort()
    real_vol_df = pd.DataFrame()

    n_obs = []
    for d in date_str_list:
        n_obs.append(close_ds[d].shape[0])
    n_obs_mode = statistics.mode(n_obs)

    for d in date_str_list:
        close_df_curr = close_ds[d]

        # Drop spurious rows with almost all nans
        close_df_curr = close_df_curr.dropna(axis=0, thresh=int(.05 * len(close_df_curr.columns)))

        # Ensure that the index is a datetime index (for time interp. of volatility)
        df_datetime_idx = pd.to_datetime(close_df_curr.index, utc=True)
        close_df_curr.set_index(pd.DatetimeIndex(df_datetime_idx), inplace=True)

        # ----- REALIZED VOLATILITY -----
        # If data are available at a fixed time step (i.e., with no missing samples),
        # use integrated volatility, otherwise use sample standard deviation
        use_high_freq_vol = True
        if use_high_freq_vol:
            real_vol_df_curr = np.log(close_df_curr)
            real_vol_df_curr = real_vol_df_curr.diff()
            real_vol_df_curr = real_vol_df_curr.dropna(how='all')
            real_vol_df_curr = real_vol_df_curr.pow(2)
            n_nans = real_vol_df_curr.isna().sum()
            penalty = 1. / pow(1. - n_nans / real_vol_df_curr.shape[0], 2)
            real_vol_df_curr = real_vol_df_curr.interpolate(method='time')
            real_vol_df_curr = real_vol_df_curr.sum()
            # Account for a different number of daily observations
            n_obs_curr = close_df_curr.shape[0]
            real_vol_df_curr = real_vol_df_curr * (n_obs_mode / n_obs_curr)
            # Apply penalty for missing values
            real_vol_df_curr = np.sqrt(real_vol_df_curr) * penalty
        else:
            real_vol_df_curr = close_df_curr.pct_change(fill_method=None)
            n_nans = real_vol_df_curr.isna().sum()
            penalty = 1. / pow(1. - n_nans / real_vol_df_curr.shape[0], 2)
            real_vol_df_curr = real_vol_df_curr.std() * penalty

        real_vol_df_curr = real_vol_df_curr.replace(0, np.nan)

        # Convert series to dataframe row
        real_vol_df_curr = real_vol_df_curr.to_frame().transpose()

        # Change datetime index to include only the day
        real_vol_df_curr.index = [d]
        real_vol_df_curr.set_index(pd.to_datetime(real_vol_df_curr.index), inplace=True)

        if real_vol_df_curr.shape[1] > 0:  # ensure that row is not empty
            real_vol_df = pd.concat([real_vol_df, real_vol_df_curr])

        if verbose:
            print(d)

    real_vol_df = reindex_by_date(real_vol_df, price_df.index)
    price_df = price_df.dropna(axis=1, how='all')
    real_vol_df = real_vol_df[price_df.columns]

    # Check for errors in the position of decimal point of last prices
    # price_df = correct_data_anomalies(price_df)

    # ----- RETURNS -----
    ret_df = price_df.pct_change(fill_method=None)
    ret_df = ret_df.dropna(axis=0, how='all')

    real_vol_df = real_vol_df.iloc[1:]  # and realized volatility

    return ret_df, real_vol_df


def correct_data_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Check for errors in position of decimal point of last prices
    col_list = df.columns
    for col in col_list:
        change_df = df[col].pct_change(fill_method=None)
        pos_jumps = (change_df > 8)
        neg_jumps = (change_df < -0.8)
        large_changes_idx = change_df[pos_jumps | neg_jumps].index

        if len(large_changes_idx) > 0:
            for _ in np.arange(2):
                # Get price level as rolling median over last 3 months
                data_level = df[col].rolling(60, min_periods=1).median()
                for idx in large_changes_idx:
                    # Detect abnormally high or low prices
                    enorm_hi_price = (df.loc[idx, col] > 8 * data_level[idx])
                    enorm_lo_price = (df.loc[idx, col] < 1 / 8 * data_level[idx])
                    if enorm_hi_price or enorm_lo_price:
                        price_ratio = df.loc[idx, col] / data_level[idx]
                        corr_factor = pow(10, -round(np.log10(abs(price_ratio))))
                        df.loc[idx, col] = corr_factor * df.loc[idx, col]

                change_df = df[col].pct_change(fill_method=None)
                pos_jumps = (change_df > 8)
                neg_jumps = (change_df < -0.8)
                large_changes_idx = change_df[pos_jumps | neg_jumps].index

    return df
