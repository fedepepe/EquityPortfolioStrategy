#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:23:10 2021

@author: federico
"""
import logging

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import os.path
import glob
import pickle
from typing import Union, Dict, Tuple
import matplotlib.pyplot as plt

from stock_picker import StockPicker
from portfolio import Portfolio
from portfolio_metrics import EnumPerfMetrics
import portfolio_plot as pf_plot
from yahoo_data_tools import ymd_date_fmt


def calculate_time(func):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        func(self, *args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'\n --- Simulation time: {time.strftime("%Mm %Ss", time.gmtime(elapsed))} --- ')

    return wrapper


class PortfolioBacktest:
    def __init__(self,
                 dataset: str,
                 n_stk: Union[int, np.ndarray] = None,
                 n_obs: Union[int, np.ndarray] = None,
                 n_reb: Union[int, np.ndarray] = None,
                 algos: Union[str, list[str]] = None,
                 wght_mtds: Union[str, list[str]] = None,
                 price_df: pd.DataFrame = None,
                 return_df: pd.DataFrame = None,
                 volat_df: pd.DataFrame = None,
                 mktcap_df: pd.DataFrame = None,
                 mkt_idx_df: pd.Series = None,
                 risk_free_ret: Union[float, pd.Series] = None,
                 endow: float = 1e6,
                 idx_start: int = None,
                 lag: int = 1,
                 nsel: float = 0.1,
                 trsctn_fee_fix: float = 0.,
                 trsctn_fee_prop: float = 10e-4,
                 risk_avers_factor: float = None,
                 max_leverage: float = 3.,
                 multi_proc: bool = True,
                 cv_opt_bw: bool = False,
                 results_dir: str = None,
                 results_tag: str = None,
                 results_date: Union[str, datetime, pd.Timestamp] = None,
                 overwrite_results: bool = False,
                 save_stk_hist: bool = False):

        self.n_stk_ar = n_stk
        if isinstance(n_stk, int):
            self.n_stk_ar = np.array([n_stk])
        self.n_stk = None
        self.n_obs_ar = n_obs
        if isinstance(n_obs, int):
            self.n_obs_ar = np.array([n_obs])
        self.n_obs = None
        self.n_reb_ar = n_reb
        if isinstance(n_reb, int):
            self.n_reb_ar = np.array([n_reb])
        self.n_reb = None
        self.algos = algos
        if isinstance(algos, str):
            self.algos = [self.algos]

        if not any([x is None for x in [n_stk, n_obs, n_reb, algos]]):
            if len(self.n_stk_ar) * len(self.n_obs_ar) * len(self.n_reb_ar) * len(self.algos) == 1:
                self.parametric_sweep = False
                if results_tag is None:
                    self.results_tag = ''
                else:
                    self.results_tag = results_tag
            else:
                self.parametric_sweep = True
                if results_tag is None:
                    self.results_tag = 'sw'
                else:
                    self.results_tag = results_tag
        else:
            self.parametric_sweep = False
            if results_tag is None:
                self.results_tag = ''
            else:
                self.results_tag = results_tag            

        self.algo = None
        self.wght_mtds = wght_mtds
        if isinstance(self.wght_mtds, str):
            self.wght_mtds = [self.wght_mtds]
        self.price_df = price_df
        self.return_df = return_df
        self.volat_df = volat_df
        self.mktcap_df = mktcap_df
        self.mkt_idx_df = mkt_idx_df
        if risk_free_ret is None:
            self.risk_free_ret = 0
            exc_ret_df = return_df
        else:
            self.risk_free_ret = risk_free_ret
            exc_ret_df = return_df.subtract(risk_free_ret, axis=0)
        # TO DO: include also the volatility of the benchmark in the Sharpe ratio formula
        self.sharpe_df = exc_ret_df / volat_df
        self.endow = endow
        if n_obs is not None:
            """ when n_obs is swept, idx_start can be set equal to the maximum n. of observations,
            to have all runs starting at the same trading day """
            if idx_start is None:
                self.idx_start = max(self.n_obs_ar) + lag - 1
            else:
                self.idx_start = max(idx_start, max(self.n_obs_ar)) + lag - 1
        self.lag = lag  # Time lag (in days) between last observation and of rebalancing
        self.nsel = nsel
        self.trsctn_fee_fix = trsctn_fee_fix
        self.trsctn_fee_prop = trsctn_fee_prop
        self.risk_avers_factor = risk_avers_factor
        self.max_leverage = max_leverage
        self.multi_proc = multi_proc
        self.cv_opt_bw = cv_opt_bw
        self.save_stk_hist = save_stk_hist

        # Initialize portfolio(s)
        self.pf_dict = None
        if wght_mtds is not None:
            self.init_portfolios(self.price_df.index[0])

        # Attributes associated to backtesting results
        if results_dir is None:
            self.results_dir = f'./{dataset}/results/'
        else:
            self.results_dir = results_dir
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self.overwrite_results = overwrite_results
        if results_date is None:
            self.results_date = datetime.today().strftime(ymd_date_fmt)
        else:
            if isinstance(results_date, pd.Timestamp):
                self.results_date = results_date.strftime(ymd_date_fmt)
            elif isinstance(results_date, datetime):
                self.results_date = results_date.strftime(ymd_date_fmt)
            elif isinstance(results_date, str):
                self.results_date = results_date
            else:
                raise Exception('Results date badly specified.')
        self.results_filenames = {}
        self.results_pickle_filename = f'{self.get_results_base_filename()}.pkl'

        # Timestamps involved into backtesting
        self.timestamps = None
        self.timestamps_reb = None

    def init_portfolios(self, timestamp):
        # Create portfolio(s)
        self.pf_dict = {}
        for wm in self.wght_mtds:
            self.pf_dict[wm] = Portfolio(self.endow)
            # Record the initial cash in the portfolio history
            self.pf_dict[wm].update_hist(timestamp)

    def get_results_base_filename(self):
        base_filename = f'{self.results_dir}{self.results_tag}_{self.results_date}'
        return base_filename

    def set_results_text_filenames(self):
        self.results_filenames = {}
        for wm in self.wght_mtds:
            self.results_filenames[wm] = f'{self.get_results_base_filename()}_{self.n_stk}_{self.algo}_{wm}.txt'
        if self.mkt_idx_df is not None:
            self.results_filenames['mkt'] = f'{self.get_results_base_filename()}_mkt.txt'

    def set_n_stk(self, n_stk: int):
        self.n_stk = n_stk
        self.set_results_text_filenames()

    def set_algo(self, algo: str):
        self.algo = algo
        self.set_results_text_filenames()

    def set_results_date(self, results_date):
        self.results_date = results_date
        self.set_results_text_filenames()

    def allocate(self,
                 timestamp: pd.Timestamp = None,
                 price_curr_dct: Dict = None) -> Dict[str, pd.Series]:
        date_time_idx = self.return_df.index

        # If no argument is passed, simply compute allocation at the end of 
        # the given time frame of observations
        if timestamp is None:
            timestamp = date_time_idx[-1]
        if price_curr_dct is None:
            price_curr_dct = self.price_df.loc[timestamp].to_dict()

        # Exclude a number 'lag' of samples before and including the current timestamp
        # and then get the last n_obs observations
        ret_df_win = self.return_df[date_time_idx <= timestamp]
        ret_df_win = ret_df_win[:-self.lag].tail(self.n_obs)
        vol_df_win = self.volat_df[date_time_idx <= timestamp]
        vol_df_win = vol_df_win[:-self.lag].tail(self.n_obs)
        sharpe_df_win = self.sharpe_df[date_time_idx <= timestamp]
        sharpe_df_win = sharpe_df_win[:-self.lag].tail(self.n_obs)

        if self.mktcap_df is None:
            cap_df_win = None
        else:
            cap_df_win = self.mktcap_df[date_time_idx <= timestamp]
            cap_df_win = cap_df_win[:-self.lag].tail(self.n_obs)

        if isinstance(self.risk_free_ret, pd.Series):
            rf_ret_df_hist = self.risk_free_ret[self.risk_free_ret.index <= timestamp][:-self.lag]
        else:
            rf_ret_df_hist = self.risk_free_ret

        # Create stock picker object
        stock_picker = StockPicker(self.n_stk, self.algo, ret_df_win, vol_df_win, sharpe_df_win,
                                   cap_df_win, self.nsel)

        # Perform stock selection
        stock_sel_df = stock_picker.pick_stocks(self.multi_proc, self.cv_opt_bw)
        stock_sel = stock_sel_df.index

        # Add column with last price
        try:
            stock_sel_df['Price'] = [price_curr_dct[tckr] for tckr in stock_sel]
        except:
            x = 0

        # Use historical returns and volatilities up to the current rebalancing day 
        # for Markowitz' portfolio optimization
        ret_sel_df_hist = self.return_df.loc[date_time_idx <= timestamp, stock_sel][:-self.lag]
        vol_sel_df_hist = self.volat_df.loc[date_time_idx <= timestamp, stock_sel][:-self.lag]

        # Compute portfolio allocation
        pf_alloc_dct = {}
        for wm in self.pf_dict:
            pf_alloc_dct[wm] = self.pf_dict[wm].compute_weights(stock_sel_df, wm,
                                                                self.risk_avers_factor,
                                                                self.max_leverage,
                                                                self.mktcap_df.loc[timestamp],
                                                                ret_sel_df_hist, vol_sel_df_hist,
                                                                rf_ret_df_hist,
                                                                self.trsctn_fee_fix,
                                                                self.trsctn_fee_prop)
        return pf_alloc_dct

    @calculate_time
    def backtest_single(self, start_date: pd.Timestamp = None, stop_date: pd.Timestamp = None):
        # Make a copy of last price dataframe that can be manipulated
        price_df = self.price_df.copy()

        # Select only data samples between start and stop date, if given
        if start_date is not None:
            price_df = price_df.loc[price_df.index >= start_date]
        if stop_date is not None:
            price_df = price_df.loc[price_df.index <= stop_date]

        # Check if there are enough samples for the first iteration
        n_samples = price_df.shape[0]
        if n_samples <= self.n_obs:
            raise Exception('Not enough data samples for backtesting.')

        # Get full list of days to be simulated and those when rebalancing occurs
        # self.idx_start is the first trading day, so the portfolios are initialized
        # at self.idx_start - 1 with the initial wealth
        self.timestamps = price_df.index[self.idx_start - 1:]
        self.timestamps_reb = price_df.index[self.idx_start::self.n_reb]

        print(f" --- #obs: {self.n_obs}, #reb: {self.n_reb}, #stk: {self.n_stk}. "
              f"Total time steps: {self.timestamps.size - 1}. --- ")

        # Build dictionary with last price for computing portfolio value
        # To this purpose, fill NaNs with last valid data, in order to
        # get an approximated value of the wealth even if not all asset prices
        # are available
        price_df_fill = price_df.fillna(method='ffill')

        # (Re-)initialize portfolio(s)
        self.init_portfolios(price_df.index[self.idx_start - 1])

        for n, timestamp in enumerate(self.timestamps[1:]):

            price_curr_dct_fill = price_df_fill.loc[timestamp].to_dict()

            for wm in self.pf_dict:
                self.pf_dict[wm].update_value(price_curr_dct_fill)

            if timestamp in self.timestamps_reb:
                # For the transaction, we use the last available prices without
                # filling NaNs, thus excluding selected stocks for which we
                # don't have a valid price
                price_curr_dct = price_df.loc[timestamp].to_dict()

                # Compute new portfolio allocation(s)
                pf_alloc_dct = self.allocate(timestamp, price_curr_dct)

                # Execute trading
                for wm in self.pf_dict:
                    self.pf_dict[wm].rebalance(pf_alloc_dct[wm], price_curr_dct,
                                               self.trsctn_fee_fix, self.trsctn_fee_prop)

            for wm in self.pf_dict:
                self.pf_dict[wm].update_hist(timestamp, self.save_stk_hist)

            if n + 1 == 5:
                print(' --- Done step', end=' ')
            if (n + 1) % 5 == 0:
                print(f'{n + 1}', end='..')

        # add market portfolio
        if self.mkt_idx_df is not None:
            self.mkt_idx_df = self.mkt_idx_df.reindex(index=self.timestamps)
            self.pf_dict['mkt'] = Portfolio(self.endow)
            self.pf_dict['mkt'].val_tot_hist = self.mkt_idx_df / self.mkt_idx_df.iloc[0] * self.endow
            self.pf_dict['mkt'].val_trans_hist = pd.Series(data=0, index=self.timestamps)

    def backtest(self, start_date: pd.Timestamp = None, stop_date: pd.Timestamp = None):
        if self.overwrite_results:
            if os.path.isfile(self.results_pickle_filename):
                os.remove(self.results_pickle_filename)
        for n_stk in self.n_stk_ar:
            for algo in self.algos:
                self.set_n_stk(n_stk=n_stk)
                self.set_algo(algo=algo)

                self.clear_result_textfiles()

                # Main loop
                start = time.perf_counter()
                n_sim = len(self.n_obs_ar) * len(self.n_reb_ar)
                n_done = 0
                for n_obs in self.n_obs_ar:
                    for n_reb in self.n_reb_ar:
                        self.n_obs = n_obs
                        self.n_reb = n_reb

                        print(f'\n --- Running sim. {n_done + 1} of {n_sim} ---')

                        self.backtest_single(start_date=start_date, stop_date=stop_date)

                        end_curr = time.perf_counter()
                        elapsed = end_curr - start  # Total time elapsed

                        # Analyze portfolio performance
                        self.analyze()

                        n_done += 1
                        elapsed_mean = elapsed / n_done

                        elapsed_str = time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed))
                        elapsed_mean_str = time.strftime('%Mm %Ss', time.gmtime(elapsed_mean))
                        print(f'\n --- Total time elapsed: {elapsed_str} '
                              f'(average sim. time: {elapsed_mean_str}) --- \n')

                        # save results to pickle file
                        self.save_results()

                        # Print results to text file in case of parameter sweep
                        self.print_results(to_file=self.parametric_sweep)

    def analyze(self):
        for wm in self.pf_dict:
            self.pf_dict[wm].analyze(mkt_ret_df=self.mkt_idx_df.pct_change(),
                                     risk_free_ret=self.risk_free_ret)

    def compute_ff_factors(self, n_factors: int):
        for wm in self.pf_dict:
            self.pf_dict[wm].compute_ff_factors(n_factors)

    def print_results(self, to_file: bool = False):
        if not to_file:
            pf_name = list(self.pf_dict.keys())[0]
            pf_metrics = self.pf_dict[pf_name].perf_metrics
            header_str = 'stk\tobs\treb\talgo\tweight'
            for metric in EnumPerfMetrics:
                if metric in pf_metrics.keys():
                    if isinstance(pf_metrics[metric].value, float):
                        header_str = f'{header_str}\t{pf_metrics[metric].label}'
            print(header_str)

        for wm, portfolio in self.pf_dict.items():
            if not portfolio.perf_metrics:
                raise Exception('Error! Run portfolio analysis first!')

            data_str = f'{self.n_stk}\t{self.n_obs}\t{self.n_reb}\t{self.algo}\t{wm}\t'
            for metric in EnumPerfMetrics:
                if metric in portfolio.perf_metrics.keys():
                    pf_metric = portfolio.perf_metrics[metric]
                    if isinstance(pf_metric.value, float):
                        if pf_metric.is_percentage:
                            data_str = f'{data_str}\t{100 * pf_metric.value:.{pf_metric.decimals}f}'
                        else:
                            data_str = f'{data_str}\t{pf_metric.value:.{pf_metric.decimals}f}'

            if to_file:
                filename = self.results_filenames[wm]
                if wm == 'mkt':
                    with open(filename, 'w') as text_file:
                        text_file.write(f'{data_str}\n')
                else:
                    with open(filename, 'a') as text_file:
                        text_file.write(f'{data_str}\n')
            else:
                print(data_str)

    def print_results_for_latex(self):
        for wm, portfolio in self.pf_dict.items():
            if not portfolio.perf_metrics:
                raise Exception('Error! Run portfolio analysis first!')

            data_str = (f'{self.algo}-{wm} & '
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.ALPHA].value:.2f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.BETA].value:.2f} & '
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.ANN_MEAN_RET].value:.2f} & '
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.VOLATILITY].value:.2f} & '
                        f'{100 * (portfolio.val_tot_hist[-1] / portfolio.val_tot_hist[0]):.2f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.MAXDD].value:.2f} & '
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.TURNOVER].value:.2f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.SHARPE].value:.4f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.SORTINO].value:.4f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.STAR].value:.4f} \\\\')
            print(data_str)

    def save_results(self):
        # load result file, if it exists already
        results_dict = self.load_results()

        # save only scalar metrics, not time series
        for wm, portfolio in self.pf_dict.items():
            perf_metrics_to_save = {}
            keys_to_save = [k for k, v in portfolio.perf_metrics.items() if isinstance(v.value, float)]
            for k in keys_to_save:
                perf_metrics_to_save[k] = portfolio.perf_metrics[k]
            results_dict[(self.n_stk, self.n_obs, self.n_reb, self.algo, wm)] = perf_metrics_to_save

        with open(self.results_pickle_filename, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_results(self, load_last: bool = False) -> Dict[Tuple, Dict]:
        if os.path.isfile(self.results_pickle_filename):
            return pd.read_pickle(self.results_pickle_filename)
        else:
            if load_last:
                logging.warning('Loading latest results saved.')
                data_file_collection = glob.glob(f'{self.results_dir}{self.results_tag}*.pkl')
                data_file_collection.sort(reverse=True)
                return pd.read_pickle(data_file_collection[0])
            else:
                return {}

    def rearrange_results(self, metric_id: EnumPerfMetrics) -> Dict[Tuple, pd.DataFrame]:
        results_dict_old = self.load_results(load_last=True)
        if len(results_dict_old) > 0:
            results_dict_new = {}
            for key, perf_dict in results_dict_old.items():
                (n_stk, n_obs, n_reb, algo, wm) = key
                if (n_stk, algo, wm) not in results_dict_new.keys():
                    results_dict_new[(n_stk, algo, wm)] = pd.DataFrame()
                results_dict_new[(n_stk, algo, wm)].loc[n_reb, n_obs] = perf_dict[metric_id].value
            return results_dict_new
        else:
            logging.warning('No valid results found.')
            return {}

    def plot_cum_wealth(self) -> plt.Figure:
        fig_wealth = None
        for wm, portfolio in self.pf_dict.items():
            if wm == 'mkt' or wm == 'market':
                linestyle = '--'
            else:
                linestyle = '-'
            fig_wealth = pf_plot.plot_cum_wealth(self.pf_dict[wm].val_tot_hist,
                                                 figure=fig_wealth,
                                                 descr=f'algo={self.algo} wght={wm}',
                                                 linestyle=linestyle)

        fig_wealth.axes[0].legend(loc='best')
        return fig_wealth

    def plot_heatmap(self, metric_id: EnumPerfMetrics) -> Dict[int, plt.Figure]:
        results_dict = self.rearrange_results(metric_id=metric_id)

        # get info on chosen performance metric
        default_pf = Portfolio()
        metric_label = default_pf.perf_metrics[metric_id].label
        reverse = default_pf.perf_metrics[metric_id].reversed_color_scale

        # iterate over all values of n_stk present in the results dictionary
        n_stk_ar = list(set([k[0] for k in results_dict.keys()]))
        figs_dict = {}
        for n_stk in n_stk_ar:
            results_dict_n_stk = {k: results_dict[k] for k in results_dict.keys() if k[0] == n_stk}
            fig_heatmap = pf_plot.plot_heatmap_mosaic(results_dict=results_dict_n_stk,
                                                      metric_label=metric_label,
                                                      reverse=reverse,
                                                      results_base_filename=self.get_results_base_filename())
            figs_dict[n_stk] = fig_heatmap
        return figs_dict

    def clear_result_textfiles(self):
        for wm in self.wght_mtds:
            file_handle = open(self.results_filenames[wm], 'w')
            file_handle.close()
