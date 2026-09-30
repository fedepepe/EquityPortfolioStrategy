#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:54:45 2021

@author: federico
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import urllib.request
import zipfile
import yahoo_data_tools
from datetime import datetime as dt

TRADING_DAYS_IN_YEAR = 252.
WEEKS_IN_YEAR = 52.


def geo_mean(iterable):
    arr = np.array(iterable) + 1.
    return arr.prod() ** (1.0 / len(arr)) - 1.


def compute_returns(cum_wealth_hist):
    returns = cum_wealth_hist.pct_change()
    returns = returns.dropna()  # Discrete returns
    return returns


def compute_exc_ret(cum_wealth_hist, returns_benchmark=0):  # Excess returns
    returns = compute_returns(cum_wealth_hist)
    if isinstance(returns_benchmark, pd.Series):
        returns_benchmark = returns_benchmark.loc[returns.index]
    exc_ret = returns - returns_benchmark
    return exc_ret


def compute_returns_volat_sharpe(cum_wealth_hist, risk_free_ret=0):
    returns = compute_returns(cum_wealth_hist)
    ret_mean = TRADING_DAYS_IN_YEAR * returns.mean()  # Mean excess return
    log_returns = np.log(1. + returns)  # Logarithmic returns
    exc_returns = compute_exc_ret(cum_wealth_hist, risk_free_ret)  # Excess returns
    exc_ret_mean = TRADING_DAYS_IN_YEAR * exc_returns.mean()  # Mean excess return
    volat = np.sqrt(TRADING_DAYS_IN_YEAR) * exc_returns.std()  # Volatility
    sharpe = exc_ret_mean / volat  # Sharpe ratio
    return returns, log_returns, exc_returns, ret_mean, exc_ret_mean, volat, sharpe


def compute_info_ratio(cum_wealth_hist, benchmark_ret):
    ret_diff = compute_exc_ret(cum_wealth_hist, benchmark_ret)
    if ret_diff.std() < 1e-4:
        return 0.
    else:
        return np.sqrt(TRADING_DAYS_IN_YEAR) * ret_diff.mean() / ret_diff.std()


def compute_sortino(cum_wealth_hist, risk_free_ret=0, min_acc_ret=0):  # Sortino ratio
    exc_ret = compute_exc_ret(cum_wealth_hist, risk_free_ret)
    exc_ret_mean = exc_ret.mean()  # Mean excess return
    exc_ret_mean = TRADING_DAYS_IN_YEAR * exc_ret_mean
    exc_ret_std_dnsd = np.sqrt(TRADING_DAYS_IN_YEAR) * exc_ret[exc_ret < min_acc_ret].std()
    return exc_ret_mean / exc_ret_std_dnsd


def compute_star(cum_wealth_hist, risk_free_ret=0, level=0.05):  # STAR ratio
    exc_ret = compute_exc_ret(cum_wealth_hist, risk_free_ret)
    exc_ret_mean = exc_ret.mean()  # Mean excess return
    value_at_risk = exc_ret.quantile(level)
    exp_tail_loss = exc_ret[exc_ret < value_at_risk].mean()
    return - exc_ret_mean / exp_tail_loss


def compute_turnover(portfolio):  # turnover
    turnover = portfolio.val_trans_hist.divide(portfolio.val_tot_hist)
    return turnover.mean()


def compute_alpha_beta(cum_wealth_hist, bema_ret_df, risk_free_ret=0):  # alpha and beta factors
    exc_ret = compute_exc_ret(cum_wealth_hist, risk_free_ret)
    bema_ret_df_win = bema_ret_df.loc[exc_ret.index]
    if isinstance(risk_free_ret, pd.Series):
        rf_ret_df_win = risk_free_ret.loc[exc_ret.index]
    else:
        rf_ret_df_win = risk_free_ret
    y = np.array(exc_ret.values, dtype=float).reshape(-1, 1)
    x = np.array(bema_ret_df_win.subtract(rf_ret_df_win, axis=0).values, dtype=float).reshape(-1, 1)
    x = sm.add_constant(x, prepend=True)
    ols = sm.OLS(y, x)
    ols_result = ols.fit()
    alpha = TRADING_DAYS_IN_YEAR * ols_result.params[0]
    beta = ols_result.params[1]
    pvalues = ols_result.pvalues
    return alpha, beta, pvalues


def compute_drawdown(cum_wealth_hist):  # drawdown
    previous_peak = cum_wealth_hist.cummax()
    drawdown = (cum_wealth_hist - previous_peak) / previous_peak
    return drawdown


def compute_ff_factors(cum_wealth_hist, n_factors, risk_free_ret=0):
    url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    file_handle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(file_handle, 'r')
    filename = zip_file_object.namelist()[0]
    file = zip_file_object.open(filename)
    factors_df = pd.read_csv(file, skiprows=3)
    factors_df = factors_df.dropna()

    n = factors_df.shape[0]
    timestamps = [dt.strptime(str(factors_df.iloc[i, 0]), '%Y%m%d') for i in range(n)]
    factors_df.iloc[:, 0] = timestamps
    timestamps = pd.to_datetime(timestamps, utc=True)
    factors_df.set_index(pd.DatetimeIndex(timestamps), inplace=True)
    factors_df.drop(factors_df.columns[0], axis=1, inplace=True)
    factors_df.drop('RF', axis=1, inplace=True)
    factors_df = factors_df / 100

    if n_factors > 3:
        url = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
        file_handle, _ = urllib.request.urlretrieve(url)
        zip_file_object = zipfile.ZipFile(file_handle, 'r')
        filename = zip_file_object.namelist()[0]
        file = zip_file_object.open(filename)
        mom_df = pd.read_csv(file, skiprows=12)
        mom_df = mom_df.dropna()

        n = mom_df.shape[0]
        timestamps = [dt.strptime(str(mom_df.iloc[i, 0]), '%Y%m%d') for i in range(n)]
        mom_df.iloc[:, 0] = timestamps
        timestamps = pd.to_datetime(timestamps, utc=True)
        mom_df.set_index(pd.DatetimeIndex(timestamps), inplace=True)
        mom_df.drop(mom_df.columns[0], axis=1, inplace=True)
        mom_df = mom_df / 100
        factors_df = pd.concat([factors_df, mom_df], axis=1)

    exc_ret = compute_exc_ret(cum_wealth_hist, risk_free_ret)

    factors_df = yahoo_data_tools.reindex_by_date(factors_df, exc_ret.index)
    factors_df = factors_df.loc[exc_ret.index]

    sel_factors_df = factors_df.iloc[:, :n_factors]
    y = np.array(exc_ret.values, dtype=float).reshape(-1, 1)
    x = np.array(sel_factors_df.values, dtype=float)
    if x.shape[1] == 1:
        x = x.reshape(-1, 1)
    x = sm.add_constant(x, prepend=True)
    ols = sm.OLS(y, x)
    max_lag = int(4 * (x.shape[0] / 100) ** (2 / 9))
    ols = ols.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
    labels = list(factors_df.columns)
    labels.insert(0, 'constant')
    ff_factors = {labels[i]: {'coeff': 100. * TRADING_DAYS_IN_YEAR * ols.params[i],
                              't_value': ols.tvalues[i],
                              'p_value': ols.pvalues[i]}
                  for i in range(n_factors + 1)}
    return ff_factors
