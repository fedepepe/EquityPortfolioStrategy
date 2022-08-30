#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""
import numpy as np
import pickle
from enum import Enum, auto
from typing import Union, List

import pandas as pd

from definitions import StockUniverses, WeightMethods, Algorithms
from portfolio import EnumPerfMetrics
from portfolio_backtest import PortfolioBacktest
from yahoo_data_downloader import YahooDataDownloader


# %% Load data
def main_download(dataset_list: list[StockUniverses]) -> None:
    for ds in dataset_list:
        yahoo_download = YahooDataDownloader(ds)
        yahoo_download.download_latest_data()


def main_download_stk_data(dataset_list: list[StockUniverses]) -> None:
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


def create_backtest_obj(dataset: StockUniverses,
                        n_stk: Union[int, np.array] = None,
                        n_obs: Union[int, np.array] = None,
                        n_reb: Union[int, np.array] = None,
                        algos: Union[Algorithms, List[Algorithms]] = None,
                        wght_mtds: Union[WeightMethods, List[WeightMethods]] = None,
                        date_start: pd.Timestamp = None,
                        date_stop: pd.Timestamp = None
                        ) -> PortfolioBacktest:
    stk_data, mkt_data = load_data(dataset)
    close_df, return_df, real_vol_df, mktcap_df, close_adj_ds = [d for d in stk_data]
    mkt_ret_df, mkt_idx_df = [d for d in mkt_data]

    if date_start is not None:
        close_df = close_df.loc[close_df.index <= date_start]

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
    if pf_backtest.parametric_sweep:
        pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.SHARPE)
        pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.IC)
    else:
        pf_backtest.plot_cum_wealth()


class UnitTests(Enum):
    RUN_DOWNLOAD_ALL = auto()
    RUN_DOWNLOAD_STOCK_DATA = auto()
    RUN_SINGLE = auto()
    RUN_SWEEP = auto()
    PLOT_RESULTS_SWEEP = auto()
    RUN_DOWNLOAD_SWEEP = auto()
    RUN_ALLOCATION = auto()


def run_unit_test(unit_test: UnitTests):
    dataset_list = [StockUniverses.SP500, StockUniverses.STOXXE600]
    wght_mtds = [WeightMethods.EQ, 
                 WeightMethods.MKTCAP, 
                 WeightMethods.RISKPAR,
                 WeightMethods.LOTP,
                 WeightMethods.ILOTP]

    if unit_test == UnitTests.RUN_DOWNLOAD_ALL:
        main_download(dataset_list=dataset_list)

    elif unit_test == UnitTests.RUN_DOWNLOAD_STOCK_DATA:
        main_download_stk_data(dataset_list=dataset_list)

    elif unit_test == UnitTests.RUN_SINGLE:
        dataset = StockUniverses.SP500
        algos = Algorithms.SEV
        n_stk = 10  # Number of stocks to hold in the portfolio
        n_obs = 90  # Number of past observations to use as training data
        n_reb = 40  # Rate of portfolio rebalancing (in trading days)
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algos, wght_mtds=wght_mtds)
        pf_backtest.backtest()
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.RUN_SWEEP:
        dataset = StockUniverses.STOXXE600
        algos = [field.value for field in Algorithms]
        n_stk = 10  # Number of stocks to hold in the portfolio
        n_obs = np.arange(20, 105, 10)  # Number of past observations to use as training data
        n_reb = np.arange(20, 65, 5)  # Rate of portfolio rebalancing (in trading days)
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algos, wght_mtds=wght_mtds)
        pf_backtest.backtest()
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.PLOT_RESULTS_SWEEP:
        dataset = StockUniverses.SP500
        pf_backtest = create_backtest_obj(dataset=dataset)
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.RUN_DOWNLOAD_SWEEP:
        # dataset_list = [StockUniverses.SP500, StockUniverses.STOXXE600]
        dataset_list = [StockUniverses.STOXXE600]
        main_download(dataset_list=dataset_list)
        algos = [field.value for field in Algorithms]
        n_stk = 10
        n_obs = np.arange(20, 105, 10)
        n_reb = np.arange(20, 65, 5)
        for dataset in dataset_list:
            pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                              algos=algos, wght_mtds=wght_mtds)
            pf_backtest.backtest()
            pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.SHARPE)
            pf_backtest.plot_heatmap(metric_id=EnumPerfMetrics.IC)

    elif unit_test == UnitTests.RUN_ALLOCATION:
        dataset = StockUniverses.STOXXE600
        algo = Algorithms.RMLIN
        wght_mtd = WeightMethods.ILOTP
        n_stk = 10  # Number of stocks to hold in the portfolio
        n_obs = 60  # Number of past observations to use as training data
        pf_backtest = create_backtest_obj(dataset=dataset, wght_mtds=wght_mtd)
        pf_backtest.n_obs = n_obs
        pf_backtest.set_n_stk(n_stk=n_stk)
        pf_backtest.set_algo(algo=algo)
        pf_alloc_dct = pf_backtest.allocate()
        df = pd.DataFrame(pf_alloc_dct[wght_mtd.value])
        df = df.reset_index()
        df = df.rename(columns={"index": "Ticker"})
        df = df.sort_values('Weight', ascending=False)
        timestamp = pf_backtest.price_df.index[-1]
        print(timestamp)
        print(df)
        date_str = timestamp.strftime("%Y_%m_%d")
        file_name = f"./predictions/{dataset}_{date_str}_{algo}_{wght_mtd}_{n_stk}_{n_obs}.xlsx"
        df.to_excel(file_name)


if __name__ == '__main__':
    unit_test = UnitTests.RUN_SINGLE
    run_unit_test(unit_test=unit_test)
