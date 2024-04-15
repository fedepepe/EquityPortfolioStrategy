#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""
import pickle
from datetime import datetime
from enum import Enum, auto
from typing import Union, List

import numpy as np
import pandas as pd

import definitions
from definitions import StockUniverses, WeightMethods, Algorithms, Strategy, TopStrategies
from portfolio import EnumPerfMetrics
from portfolio_backtest import PortfolioBacktest
from yahoo_data_downloader import YahooDataDownloader

WGHT_METHODS_ALL = [w.value for w in WeightMethods]


# %% Load data
def main_download(dataset_list: List[StockUniverses] = (StockUniverses.SP500, StockUniverses.STOXXE600)):
    for ds in dataset_list:
        yahoo_download = YahooDataDownloader(ds)
        yahoo_download.download_latest_data()


def main_download_stk_data(dataset_list: List[StockUniverses] = (StockUniverses.SP500, StockUniverses.STOXXE600)):
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
                        date_start: datetime = None,
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
                                    bema_idx_df=mkt_idx_df, risk_free_ret=None, date_start=date_start,
                                    endow=1e4, lag=data_lag,
                                    trx_fee_fix=trsctn_fee_fix, trx_fee_prop=trsctn_fee_prop,
                                    risk_avers_factor=None,
                                    multi_proc=False, cv_opt_bw=False, save_stk_hist=True,
                                    output_figs_format='png')
    return pf_backtest


def run_sweep(strategies: List[Strategy]):
    date_start = pd.to_datetime('2021-02-01')
    n_reb = np.arange(20, 65, 5)
    for strategy in strategies:
        dataset = strategy.dataset
        algos = strategy.algo
        wght_mtds = strategy.wght_mtds
        n_stk = strategy.n_stk
        n_obs = strategy.n_obs
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algos, wght_mtds=wght_mtds, date_start=date_start)
        pf_backtest.backtest_sweep_start()
        plot_results(pf_backtest=pf_backtest)


def plot_results(pf_backtest: PortfolioBacktest):
    if pf_backtest.parametric_sweep:
        pf_backtest.plot_heatmap(metric=EnumPerfMetrics.SHARPE)
        pf_backtest.plot_heatmap(metric=EnumPerfMetrics.IC)
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
    strategies = [
        # Strategy(dataset=StockUniverses.SP500, algo=Algorithms.SEV,
        #          wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
        # Strategy(dataset=StockUniverses.SP500, algo=Algorithms.RMSEV,
        #          wght_mtds=WGHT_METHODS_ALL,
        #          n_stk=20, n_obs=np.arange(20, 95, 10)),
        Strategy(dataset=StockUniverses.SP500, algo=Algorithms.MAX_SHARPE,
                 wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
        Strategy(dataset=StockUniverses.SP500, algo=Algorithms.LOLIN,
                 wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
        # Strategy(dataset=StockUniverses.STOXXE600, algo=Algorithms.LORMLIN,
        #          wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
        # Strategy(dataset=StockUniverses.STOXXE600, algo=Algorithms.SEV,
        #          wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
        # Strategy(dataset=StockUniverses.STOXXE600, algo=Algorithms.RMSEV,
        #          wght_mtds=WGHT_METHODS_ALL, n_stk=10, n_obs=np.arange(20, 95, 10)),
    ]

    if unit_test == UnitTests.RUN_DOWNLOAD_ALL:
        main_download()

    elif unit_test == UnitTests.RUN_DOWNLOAD_STOCK_DATA:
        main_download_stk_data()

    elif unit_test == UnitTests.RUN_SINGLE:
        # strategy = TopStrategies.SP500.value
        strategy = definitions.Strategy(dataset=StockUniverses.SP500,
                                        algo=Algorithms.LOLIN,
                                        wght_mtds=WeightMethods.RISKPAR,
                                        n_stk=2,
                                        n_obs=60)
        dataset = strategy.dataset
        algo = strategy.algo
        wght_mtd = strategy.wght_mtds
        n_stk = strategy.n_stk
        n_obs = strategy.n_obs
        n_reb = 10
        date_start = None  # pd.to_datetime('2021-01-01')
        pf_backtest = create_backtest_obj(dataset=dataset, n_stk=n_stk, n_obs=n_obs, n_reb=n_reb,
                                          algos=algo, wght_mtds=wght_mtd, date_start=date_start)
        pf_backtest.backtest()
        plot_results(pf_backtest=pf_backtest)
        pf_alloc_dct = pf_backtest.allocate()
        df = pd.DataFrame(pf_alloc_dct[wght_mtd])
        df = df.reset_index()
        df = df.rename(columns={"index": "Ticker"})
        df = df.sort_values('Weight', ascending=False)
        timestamp = pf_backtest.price_df.index[-1]
        print(timestamp)
        print(df)

    elif unit_test == UnitTests.RUN_SWEEP:
        run_sweep(strategies=strategies)

    elif unit_test == UnitTests.PLOT_RESULTS_SWEEP:
        dataset = StockUniverses.SP500
        pf_backtest = create_backtest_obj(dataset=dataset)
        plot_results(pf_backtest=pf_backtest)

    elif unit_test == UnitTests.RUN_DOWNLOAD_SWEEP:
        main_download()
        run_sweep(strategies=strategies)

    elif unit_test == UnitTests.RUN_ALLOCATION:
        for strategy in TopStrategies:
            dataset = strategy.value.dataset
            algo = strategy.value.algo
            wght_mtd = strategy.value.wght_mtd
            n_stk = strategy.value.n_stk
            n_obs = strategy.value.n_obs
            pf_backtest = create_backtest_obj(dataset=dataset, wght_mtds=wght_mtd)
            pf_backtest.n_obs = n_obs
            pf_backtest.set_n_stk(n_stk=n_stk)
            pf_backtest.set_algo(algo=algo)
            pf_alloc_dct = pf_backtest.allocate()
            df = pd.DataFrame(pf_alloc_dct[wght_mtd])
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
