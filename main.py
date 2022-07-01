#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
import pickle
from enum import Enum
from typing import Union

from portfolio import EnumPerfMetrics
from portfolio_backtest import PortfolioBacktest
from yahoo_data_downloader import YahooDataDownloader


# %% Load data
def main_download(dataset_list: list[str]) -> None:
    for ds in dataset_list:
        yahoo_download = YahooDataDownloader(ds)
        yahoo_download.download_latest_data()


def main_download_stk_data(dataset_list: list[str]) -> None:
    from yahoo_data_tools import get_tickers
    for ds in dataset_list:
        yahoo_download = YahooDataDownloader(ds)
        tickers = get_tickers(ds)
        yahoo_download.download_stk_data(tickers=tickers)


def load_data(dataset):
    yahoo_download = YahooDataDownloader(dataset)
    stk_data = yahoo_download.get_stock_data()
    return_df = stk_data[1]
    mkt_data = yahoo_download.get_mkt_data(return_df.index)
    return stk_data, mkt_data


def create_backtest_obj(dataset: str,
                        n_stk: Union[int, np.array] = None,
                        n_obs: Union[int, np.array] = None,
                        n_reb: Union[int, np.array] = None,
                        algos: Union[str, list[str]] = None,
                        wght_mtds: Union[str, list[str]] = None
                        ) -> PortfolioBacktest:
    stk_data, mkt_data = load_data(dataset)
    close_df, return_df, real_vol_df, mktcap_df, close_adj_ds = [d for d in stk_data]
    mkt_ret_df, mkt_idx_df = [d for d in mkt_data]

    # Load backtesting parameters
    with open('parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)

    trsctn_fee_fix = parameters[dataset]['trsctn_fee_fix']  # Fixed transaction fees
    trsctn_fee_prop = parameters[dataset]['trsctn_fee_prop']  # Proportional transaction fees
    data_lag = parameters[dataset]['data_lag']

    pf_backtest = PortfolioBacktest(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                    algos=algos, wght_mtds=wght_mtds, price_df=close_df,
                                    return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                    bema_idx_df=mkt_idx_df, risk_free_ret=None,
                                    idx_start=40, lag=data_lag,
                                    trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                    risk_avers_factor=None,
                                    multi_proc=True, cv_opt_bw=False, save_stk_hist=False,
                                    output_figs_format='png')
    return pf_backtest


def plot_results(pf_backtest: PortfolioBacktest):
    if len(pf_backtest.n_obs_ar) * len(pf_backtest.n_reb_ar) == 1:
        pf_backtest.plot_cum_wealth()
    else:
        pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.IC)


class UnitTests(Enum):
    RUN_DOWNLOAD_ALL = 0
    RUN_DOWNLOAD_STOCK_DATA = 1
    RUN_SINGLE = 2
    RUN_SWEEP = 3
    PLOT_RESULTS_SWEEP = 4
    RUN_DOWNLOAD_SWEEP = 5


def run_unit_test(unit_test: UnitTests):
    dataset_list = ['SP500', 'STOXXE600']
    wght_mtds = ['equal', 'mktcap', 'riskpar', 'lotp', 'ilotp']

    if unit_test == UnitTests.RUN_DOWNLOAD_ALL:
        main_download(dataset_list=dataset_list)

    elif unit_test == UnitTests.RUN_DOWNLOAD_STOCK_DATA:
        main_download_stk_data(dataset_list=dataset_list)

    elif unit_test == UnitTests.RUN_SINGLE:
        dataset = 'STOXXE600'
        algos = 'lolin'
        n_stk = 20  # Number of stocks to hold in the portfolio
        n_obs = 40  # Number of past observations to use as training data
        n_reb = 100  # Rate of portfolio rebalancing (in trading days)
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algos, wght_mtds=wght_mtds)
        pf_backtest.backtest()
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.RUN_SWEEP:
        dataset = 'SP500'
        algos = ['sev', 'rmsev', 'lin', 'rmlin']
        n_stk = 10  # Number of stocks to hold in the portfolio
        n_obs = np.arange(15, 65, 5)  # Number of past observations to use as training data
        n_reb = np.arange(15, 65, 5)  # Rate of portfolio rebalancing (in trading days)
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algos, wght_mtds=wght_mtds)
        pf_backtest.backtest()
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.PLOT_RESULTS_SWEEP:
        for dataset in dataset_list:
            pf_backtest = create_backtest_obj(dataset=dataset)
            plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.RUN_DOWNLOAD_SWEEP:
        dataset_list = ['SP500', 'STOXXE600']
        # main_download(dataset_list=dataset_list)
        # algos = ['sev', 'rmsev', 'lolin', 'lormlin', 'mtm', 'rmmtm']
        algos = ['lin', 'rmlin', 'lolin', 'lormlin']
        n_stk = 20
        n_obs = np.arange(15, 65, 5)
        n_reb = np.arange(15, 65, 5)
        for dataset in dataset_list:
            pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                              algos=algos, wght_mtds=wght_mtds)
            pf_backtest.backtest()
            pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.SHARPE)
            pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.IC)


if __name__ == '__main__':
    unit_test = UnitTests.RUN_SINGLE
    run_unit_test(unit_test=unit_test)
