#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:23:10 2021

@author: federico
"""
import glob
import logging
import os.path
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import portfolio_plot as pf_plot
from definitions import StockUniverses, Algorithms, WeightMethods
from portfolio import Portfolio
from portfolio_metrics import EnumPerfMetrics
from stock_picker import StockPicker
from yahoo_data_tools import YMD_DATE_FORMAT

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'\n --- Simulation time: {time.strftime("%Mm %Ss", time.gmtime(elapsed))} --- ')

    return wrapper


class PortfolioBacktest:
    def __init__(self,
                 dataset: StockUniverses,
                 n_stk: Union[int, np.ndarray] = None,
                 n_obs: Union[int, np.ndarray] = None,
                 n_reb: Union[int, np.ndarray] = None,
                 algos: Union[Algorithms, List[Algorithms]] = None,
                 wght_mtds: Union[WeightMethods, List[WeightMethods]] = None,
                 price_df: pd.DataFrame = None,
                 return_df: pd.DataFrame = None,
                 volat_df: pd.DataFrame = None,
                 mktcap_df: pd.DataFrame = None,
                 bema_idx_df: pd.Series = None,
                 risk_free_ret: Union[float, pd.Series] = None,
                 date_start: pd.Timestamp = None,
                 date_stop: pd.Timestamp = None,
                 endow: float = 1e6,
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
                 save_stk_hist: bool = False,
                 output_figs_format: str = True):

        self.dataset = dataset
        self.n_stk_ar = n_stk
        if isinstance(n_stk, int):
            self.n_stk_ar = np.array([n_stk])
        self.n_obs_ar = n_obs
        if isinstance(n_obs, int):
            self.n_obs_ar = np.array([n_obs])
        self.n_reb_ar = n_reb
        if isinstance(n_reb, int):
            self.n_reb_ar = np.array([n_reb])
        self.algos = algos
        if isinstance(algos, str):
            self.algos = [self.algos]

        self.n_stk, self.n_obs, self.n_reb = None, None, None
        self.algo = None

        self.results_dir = results_dir
        self.results = {}
        self.results_tag = results_tag

        self.wght_mtds = wght_mtds
        self.price_df = price_df
        self.return_df = return_df
        self.volat_df = volat_df
        self.mktcap_df = mktcap_df
        self.bema_idx_df = bema_idx_df
        self.risk_free_ret = risk_free_ret
        self.endow = endow

        self.pf_dict = {}

        self.date_start = date_start
        self.date_stop = date_stop

        # timestamps involved into backtesting
        self.timestamps = None
        self.timestamps_reb = None

        self.lag = lag  # time lag (in days) between last observation and of rebalancing
        self.nsel = nsel
        self.trsctn_fee_fix = trsctn_fee_fix
        self.trsctn_fee_prop = trsctn_fee_prop
        self.risk_avers_factor = risk_avers_factor
        self.max_leverage = max_leverage
        self.multi_proc = multi_proc
        self.cv_opt_bw = cv_opt_bw
        self.save_stk_hist = save_stk_hist

        # attributes associated to backtesting results
        self.overwrite_results = overwrite_results
        self.results_date = results_date
        self.results_filenames = {}
        self.output_figs_format = output_figs_format

        self.__post_init__()

    def __post_init__(self):
        if self.risk_free_ret is None:
            self.risk_free_ret = 0
            exc_ret_df = self.return_df
        else:
            exc_ret_df = self.return_df.subtract(self.risk_free_ret, axis=0)
        # TO DO: include also the volatility of the benchmark in the Sharpe ratio formula
        self.sharpe_df = exc_ret_df / self.volat_df

        if self.results_dir is None:
            self.results_dir = f'./{self.dataset}/results/'
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        if not any([x is None for x in [self.n_stk_ar, self.n_obs_ar, self.n_reb_ar, self.algos]]):
            if len(self.n_stk_ar) * len(self.n_obs_ar) * len(self.n_reb_ar) * len(self.algos) == 1:
                self.parametric_sweep = False
                if self.results_tag is None:
                    self.results_tag = ''
            else:
                self.parametric_sweep = True
                if self.results_tag is None:
                    self.results_tag = 'sw'
        else:
            self.load_results()
            if len(self.results) == 1:
                self.parametric_sweep = False
            else:
                self.parametric_sweep = True
            self.results_tag = ''

        if self.date_start is not None:
            self.idx_start = (self.price_df.index.tz_localize(None) < self.date_start).sum() + self.lag - 1

        elif self.n_obs_ar is not None:
            # when n_obs is swept, idx_start can be set equal to the maximum n. of observations,
            # to have all runs starting at the same trading day
            self.idx_start = max(self.n_obs_ar) + self.lag - 1

        if isinstance(self.wght_mtds, str):
            self.wght_mtds = [self.wght_mtds]

        # initialize portfolio(s)
        if self.wght_mtds is not None:
            self.init_portfolios(self.price_df.index[0])

        if self.results_date is None:
            self.results_date = datetime.today().strftime(YMD_DATE_FORMAT)
        else:
            if isinstance(self.results_date, pd.Timestamp):
                self.results_date = self.results_date.strftime(YMD_DATE_FORMAT)
            elif isinstance(self.results_date, datetime):
                self.results_date = self.results_date.strftime(YMD_DATE_FORMAT)
            else:
                raise Exception('Results date badly specified.')

        self.results_pickle_filename = f'{self.get_results_base_filename()}.pkl'

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
        if self.bema_idx_df is not None:
            self.results_filenames['mkt'] = f'{self.get_results_base_filename()}_mkt.txt'

    def set_n_stk(self, n_stk: int):
        self.n_stk = n_stk
        self.set_results_text_filenames()

    def set_algo(self, algo: Union[Algorithms, List[Algorithms]]):
        self.algo = algo
        self.set_results_text_filenames()

    def set_results_date(self, results_date):
        self.results_date = results_date
        self.set_results_text_filenames()

    def allocate(self,
                 timestamp: pd.Timestamp = None,
                 price_curr_dct: Dict = None) -> Dict[str, pd.Series]:
        date_time_idx = self.return_df.index

        # if no argument is passed, simply compute allocation at the end of the given time frame of observations
        if timestamp is None:
            timestamp = date_time_idx[-1]
        if price_curr_dct is None:
            price_curr_dct = self.price_df.loc[timestamp].to_dict()

        # exclude 'lag' samples before and including the current timestamp and then get the last n_obs observations
        ret_df_win = self.return_df[date_time_idx <= timestamp - timedelta(self.lag)].tail(self.n_obs)
        if len(ret_df_win) < self.n_obs:
            logger.warning('Not enough samples for the given rolling window length.')
        vol_df_win = self.volat_df[date_time_idx <= timestamp - timedelta(self.lag)].tail(self.n_obs)
        sharpe_df_win = self.sharpe_df[date_time_idx <= timestamp - timedelta(self.lag)].tail(self.n_obs)

        if self.mktcap_df is None:
            mktcap_df_win = None
        else:
            mktcap_df_win = self.mktcap_df[date_time_idx <= timestamp - timedelta(self.lag)].tail(self.n_obs)

        if isinstance(self.risk_free_ret, pd.Series):
            rf_ret_df_hist = self.risk_free_ret[date_time_idx <= timestamp - timedelta(self.lag)]
        else:
            rf_ret_df_hist = self.risk_free_ret

        # create stock picker object
        stock_picker = StockPicker(n_stk=self.n_stk,
                                   algo=self.algo,
                                   ret_df=ret_df_win,
                                   vol_df=vol_df_win,
                                   sharpe_df=sharpe_df_win,
                                   mktcap_df=mktcap_df_win,
                                   nsel=self.nsel)

        # perform stock selection
        stock_sel_df = stock_picker.pick_stocks(multi_proc=self.multi_proc, cv_opt_bw=self.cv_opt_bw)
        stock_sel = stock_sel_df.index

        # Add column with last price
        stock_sel_df['Price'] = [price_curr_dct[tkr] for tkr in stock_sel]

        # use historical returns and volatilities up to the current rebalancing day
        # for Markowitz' portfolio optimization
        ret_sel_df_hist = self.return_df.loc[date_time_idx <= timestamp, stock_sel][:-self.lag]
        vol_sel_df_hist = self.volat_df.loc[date_time_idx <= timestamp, stock_sel][:-self.lag]

        # compute portfolio allocation
        pf_alloc_dct = {}
        for wm in self.pf_dict:
            pf_alloc_dct[wm] = self.pf_dict[wm].compute_weights(stock_sel_df=stock_sel_df,
                                                                wght_mtd=wm,
                                                                risk_avers=self.risk_avers_factor,
                                                                max_lvrg=self.max_leverage,
                                                                mktcap_df=self.mktcap_df.loc[timestamp],
                                                                ret_df=ret_sel_df_hist,
                                                                vol_df=vol_sel_df_hist,
                                                                bema_ret_df=rf_ret_df_hist,
                                                                trsctn_fee_fix=self.trsctn_fee_fix,
                                                                trsctn_fee_prop=self.trsctn_fee_prop)
        return pf_alloc_dct

    @calculate_time
    def backtest_single(self, offset_start: int = 0):
        # adjust starting index
        idx_start = self.idx_start + offset_start

        # build index of backtesting timestamps
        timestamps = self.price_df.index

        # select only data samples up to stop date, if given
        if self.date_stop is not None:
            timestamps = timestamps[timestamps.tz_localize(None) <= self.date_stop]

        # check if there are enough samples for the first iteration
        if len(timestamps) <= self.n_obs:
            raise Exception('Not enough data samples for backtesting.')

        # get full list of days for backtesting and those when rebalancing occurs
        # self.idx_start is the first trading day, so the portfolios are initialized
        # at self.idx_start - 1 with the initial wealth
        self.timestamps = timestamps[idx_start - 1:]
        self.timestamps_reb = timestamps[idx_start::self.n_reb]

        print(f" --- Algo: {self.algo}, #obs: {self.n_obs}, #reb: {self.n_reb}, #stk: {self.n_stk}. "
              f"Total time steps: {self.timestamps.size - 1}. --- ")

        # build dictionary with last price for computing portfolio value. To this purpose, fill NaNs with last valid
        # data, in order to get an approximated value of the wealth even if not all asset prices are available
        price_df_fill = self.price_df.fillna(method='ffill')

        # (re-)initialize portfolio(s)
        self.init_portfolios(timestamps[idx_start - 1])

        for n, timestamp in enumerate(self.timestamps[1:]):
            price_curr_dct_fill = price_df_fill.loc[timestamp].to_dict()

            for wm in self.pf_dict:
                self.pf_dict[wm].update_value(price_curr_dct_fill)

            if timestamp in self.timestamps_reb:
                # for the transaction, we use the last available prices without filling NaNs, thus excluding
                # selected stocks for which we don't have a valid price
                price_curr_dct = self.price_df.loc[timestamp].to_dict()

                # compute new portfolio allocation(s)
                pf_alloc_dct = self.allocate(timestamp, price_curr_dct)

                # execute trading
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
        if self.bema_idx_df is not None:
            bema_idx_df = self.bema_idx_df.reindex(index=self.timestamps)
            self.pf_dict['mkt'] = Portfolio(self.endow)
            self.pf_dict['mkt'].val_tot_hist = bema_idx_df / bema_idx_df.iloc[0] * self.endow
            self.pf_dict['mkt'].val_trans_hist = pd.Series(data=0, index=self.timestamps)

    def backtest(self):
        if self.overwrite_results:
            if os.path.isfile(self.results_pickle_filename):
                os.remove(self.results_pickle_filename)
        for n_stk in self.n_stk_ar:
            for algo in self.algos:
                self.set_n_stk(n_stk=n_stk)
                self.set_algo(algo=algo)

                if self.parametric_sweep:
                    self.clear_result_text_files()

                # main loop
                start = time.perf_counter()
                n_sim = len(self.n_obs_ar) * len(self.n_reb_ar)
                n_done = 0
                for n_obs in self.n_obs_ar:
                    for n_reb in self.n_reb_ar:
                        self.n_obs = n_obs
                        self.n_reb = n_reb

                        print(f'\n --- Running backtesting {n_done + 1} of {n_sim} ---')

                        self.backtest_single()

                        end_curr = time.perf_counter()
                        elapsed = end_curr - start  # Total time elapsed

                        # analyze portfolio performance
                        self.analyze()

                        n_done += 1
                        elapsed_mean = elapsed / n_done

                        elapsed_str = time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed))
                        elapsed_mean_str = time.strftime('%Mm %Ss', time.gmtime(elapsed_mean))
                        print(f'\n --- Total time elapsed: {elapsed_str} '
                              f'(average simulation time: {elapsed_mean_str}) --- \n')

                        self.save_results()

                        # Print results to text file in case of parameter sweep
                        self.print_results(to_file=self.parametric_sweep)

                # save results to pickle file
                if self.parametric_sweep:
                    self.save_pickle_results()

    def backtest_sweep_start(self):
        if self.overwrite_results:
            if os.path.isfile(self.results_pickle_filename):
                os.remove(self.results_pickle_filename)
        n_stk = self.n_stk_ar[0]
        self.set_n_stk(n_stk=n_stk)
        for n_obs in self.n_obs_ar:
            for n_reb in self.n_reb_ar:
                self.n_obs = n_obs
                self.n_reb = n_reb

                for algo in self.algos:
                    self.set_algo(algo=algo)

                    # main loop
                    start = time.perf_counter()
                    n_sim = 30
                    n_done = 0
                    weekly_returns_df = pd.DataFrame()
                    offset_start = 0
                    for _ in range(n_sim):
                        print(f'\n --- Running backtesting {n_done + 1} of {n_sim} ---')

                        self.backtest_single(offset_start=offset_start)

                        end_curr = time.perf_counter()
                        elapsed = end_curr - start  # Total time elapsed

                        # analyze portfolio performance
                        # self.analyze()

                        n_done += 1
                        elapsed_mean = elapsed / n_done

                        elapsed_str = time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed))
                        elapsed_mean_str = time.strftime('%Mm %Ss', time.gmtime(elapsed_mean))
                        print(f'\n --- Total time elapsed: {elapsed_str} '
                              f'(average simulation time: {elapsed_mean_str}) --- \n')

                        rets_df = pd.DataFrame()
                        for wm, portfolio in self.pf_dict.items():
                            rets_df[wm] = portfolio.val_tot_hist.resample('W').last().pct_change().dropna()
                        weekly_returns_df = pd.concat([weekly_returns_df, rets_df], axis=0)

                        offset_start += 2

                    for wm in self.pf_dict.keys():
                        if wm == 'mkt':
                            continue
                        y = np.array(weekly_returns_df[wm].values, dtype=float).reshape(-1, 1)
                        x = np.array(weekly_returns_df['mkt'].values, dtype=float).reshape(-1, 1)
                        ols = sm.OLS(y, sm.add_constant(x, prepend=True))
                        ols_result = ols.fit()
                        alpha = ols_result.params[0]
                        beta = ols_result.params[1]
                        pvalue = ols_result.pvalues[0]

                        results_str = (f'{self.n_stk}\t{self.n_obs}\t{self.n_reb}\t{algo:<7}\t{wm:<7}\t'
                                       f'alpha: {52. * alpha:.2%}, pval: {pvalue:.3f}, beta: {beta:.2f}')
                        print(results_str)

                        filename = f'{self.get_results_base_filename()}_{self.n_stk}_{self.algo}_{wm}_ic.txt'
                        with open(filename, 'a') as text_file:
                            text_file.write(f'{results_str}\n')

                    # fig, ax = plt.subplots(figsize=(9, 9))
                    # ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")
                    # xseq = np.linspace(min(x), max(x), num=100)
                    # ax.plot(xseq, alpha + beta * xseq, color="k", lw=2.5)
                    # plt.show()

    def analyze(self):
        for wm in self.pf_dict:
            self.pf_dict[wm].analyze(mkt_ret_df=self.bema_idx_df.pct_change(),
                                     risk_free_ret=self.risk_free_ret)

    def compute_ff_factors(self, n_factors: int):
        for wm in self.pf_dict:
            self.pf_dict[wm].compute_ff_factors(n_factors)

    def print_results(self, to_file: bool = False):
        if not to_file:
            pf_name = list(self.pf_dict.keys())[0]
            pf_metrics = self.pf_dict[pf_name].perf_metrics
            header_str = f'stk\tobs\treb\talgo\tweight  '
            for metric in EnumPerfMetrics:
                if metric in pf_metrics.keys():
                    if isinstance(pf_metrics[metric].value, float):
                        header_str = f'{header_str}\t{pf_metrics[metric].label}'
            print(header_str)

        for wm, portfolio in self.pf_dict.items():
            if not portfolio.perf_metrics:
                raise Exception('Error! Run portfolio analysis first!')

            if wm == 'mkt':
                results_str = f'{self.n_stk}\t{self.n_obs}\t{self.n_reb}\t{wm:<7}\t{wm:<7}\t'
            else:
                results_str = f'{self.n_stk}\t{self.n_obs}\t{self.n_reb}\t{self.algo:<7}\t{wm:<7}\t'

            for metric in EnumPerfMetrics:
                if metric in portfolio.perf_metrics.keys():
                    pf_metric = portfolio.perf_metrics[metric]
                    if np.isnan(pf_metric.value):
                        continue
                    if isinstance(pf_metric.value, float):
                        results_str = results_str + f'\t{pf_metric.value:{pf_metric.format}}'

            if to_file:
                filename = self.results_filenames[wm]
                if wm == 'mkt':
                    with open(filename, 'w') as text_file:
                        text_file.write(f'{results_str}\n')
                else:
                    with open(filename, 'a') as text_file:
                        text_file.write(f'{results_str}\n')
            else:
                print(results_str)

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
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.MAXDD].value:.2f} & '
                        f'{100 * portfolio.perf_metrics[EnumPerfMetrics.TURNOVER].value:.2f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.SHARPE].value:.4f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.SORTINO].value:.4f} & '
                        f'{portfolio.perf_metrics[EnumPerfMetrics.STAR].value:.4f} \\\\')
            print(data_str)

    def save_results(self):
        # save only scalar metrics, not time series
        for wm, portfolio in self.pf_dict.items():
            perf_metrics_to_save = {}
            keys_to_save = [k for k, v in portfolio.perf_metrics.items() if isinstance(v.value, float)]
            for k in keys_to_save:
                perf_metrics_to_save[k] = portfolio.perf_metrics[k]
            self.results[(self.n_stk, self.n_obs, self.n_reb, self.algo, wm)] = perf_metrics_to_save

    def save_pickle_results(self):
        with open(self.results_pickle_filename, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_results(self, results_filename: str = None):
        if hasattr(self, 'results_pickle_filename'):
            if os.path.isfile(self.results_pickle_filename):
                results = pd.read_pickle(self.results_pickle_filename)
            else:  # load last available results
                data_file_collection = glob.glob(f'{self.results_dir}*.pkl')
                data_file_collection.sort(reverse=True)
                last_results_filename = data_file_collection[0]
                logger.info(f'Loading latest results saved: {last_results_filename}.')
                results = pd.read_pickle(last_results_filename)
        else:
            if results_filename is not None:
                full_results_path = glob.glob(f'{self.results_dir}{results_filename}')
                logger.info(f'Loading results from {full_results_path}.')
                results = pd.read_pickle(full_results_path)
            else:  # load last available results
                data_file_collection = glob.glob(f'{self.results_dir}*.pkl')
                data_file_collection.sort(reverse=True)
                last_results_filename = data_file_collection[0]
                logger.info(f'Loading latest results saved: {last_results_filename}.')
                results = pd.read_pickle(last_results_filename)
        self.results = results

    def rearrange_results(self, metric_id: EnumPerfMetrics) -> Dict[Tuple, pd.DataFrame]:
        if len(self.results) > 0:
            results_dict_new = {}
            for key, perf_dict in self.results.items():
                (n_stk, n_obs, n_reb, algo, wm) = key
                if (n_stk, algo, wm) not in results_dict_new.keys():
                    results_dict_new[(n_stk, algo, wm)] = pd.DataFrame()
                results_dict_new[(n_stk, algo, wm)].loc[n_reb, n_obs] = perf_dict[metric_id].value
            return results_dict_new
        else:
            logger.warning('No valid results found.')
            return {}

    def plot_cum_wealth(self) -> plt.Figure:
        fig_wealth = None
        for wm, portfolio in self.pf_dict.items():
            if wm == 'mkt' or wm == 'market':
                linestyle = '--'
                algo = 'market'
            else:
                linestyle = '-'
                algo = f'algo={self.algo} wght={wm}'
            fig_wealth = pf_plot.plot_cum_wealth(self.pf_dict[wm].val_tot_hist,
                                                 figure=fig_wealth,
                                                 descr=algo,
                                                 linestyle=linestyle)

        fig_wealth.axes[0].legend(loc='best')
        fig_wealth.suptitle(str(self.dataset))
        fig_wealth.show()
        return fig_wealth

    def plot_heatmap(self, metric_id: EnumPerfMetrics) -> Dict[int, plt.Figure]:
        results_dict = self.rearrange_results(metric_id=metric_id)

        # get info on chosen performance metric
        default_pf = Portfolio()
        metric_label = default_pf.perf_metrics[metric_id].label
        reverse = default_pf.perf_metrics[metric_id].rev_color_scale

        # iterate over all values of n_stk present in the results dictionary
        n_stk_ar = list(set([k[0] for k in results_dict.keys()]))
        figs_dict = {}
        for n_stk in n_stk_ar:
            results_dict_n_stk = {k: results_dict[k] for k in results_dict.keys() if k[0] == n_stk}
            fig_heatmap = pf_plot.plot_heatmap_mosaic(results_dict=results_dict_n_stk,
                                                      title=f'{self.dataset} - {metric_label}',
                                                      reverse=reverse)
            figs_dict[n_stk] = fig_heatmap
            if self.output_figs_format == 'png':
                fig_name = f'{self.get_results_base_filename()}_{n_stk}_{metric_label}.png'
                fig_heatmap.savefig(fig_name, dpi=600)
            elif self.output_figs_format == 'pdf':
                fig_name = f'{self.get_results_base_filename()}_{n_stk}_{metric_label}.pdf'
                fig_heatmap.savefig(fig_name, format='pdf', bbox_inches='tight')
        return figs_dict

    def clear_result_text_files(self):
        for wm in self.wght_mtds:
            file_handle = open(self.results_filenames[wm], 'w')
            file_handle.close()
