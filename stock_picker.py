#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import LeaveOneOut
import portfolio_optimization as mv_opt
# import statsmodels.api as sm
from scipy import stats
import multiprocessing
import warnings
import sys


def gau_ker(x):
    return 1. / np.sqrt(2 * np.pi) * np.exp(- 0.5 * pow(x, 2))


def kernel_density(x, x_pts, bw):
    """
    x:     vector of input samples
    x_pts: vector of values where the density is to be calculated
    bw:    kernel bandwidth """
    n_obs = len(x)
    fx_hat = np.nan * x_pts
    for i in np.arange(len(x_pts)):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
    return fx_hat


def kernel_density_support(x, bw):
    n_obs = len(x)
    grid_size = int(2 ** np.ceil(np.log2(10. * n_obs)))
    cut = 3.0
    a = np.min(x) - cut * bw
    b = np.max(x) + cut * bw
    x_pts, dx = np.linspace(a, b, grid_size, retstep=True)
    return x_pts, dx


def nada_wats_estim(x, y, x_pts, bw):
    fx_hat, phi_hat = np.nan * x_pts, np.nan * x_pts
    n_obs, n_pts = len(x), len(x_pts)
    for i in np.arange(n_pts):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
        phi_hat[i] = 1. / n_obs * sum(ker_x * y)

    fx_hat[fx_hat == 0] = sys.float_info.min  # Replace zeros with minimum float number

    # Conditional expectation of Y given X, i.e., regression function
    # a.k.a. Nadaraya-Watson estimator
    g_hat = phi_hat / fx_hat
    return g_hat, fx_hat


def get_bw_scott(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get a robust estimate of sigma
        sig = np.nanmedian(np.abs(x - np.nanmedian(x))) / 0.6745

    if sig <= 0:
        sig = max(x) - min(x)

    if sig > 0:
        n = len(x)
        return 1.059 * sig * n ** (-0.2)
    else:
        return np.nan


def get_optimum_cv_bandwidth(x, bw_ref, n_bw_points=21):
    bw_vec = np.logspace(np.log10(bw_ref / 10.), np.log10(bw_ref * 10.), n_bw_points)
    n_obs = len(x)

    emp_risk = np.array([0.0] * len(bw_vec))
    emp_risk_1 = np.array([0.0] * len(bw_vec))
    emp_risk_2 = np.array([0.0] * len(bw_vec))

    i = 0
    for bw in bw_vec:
        x_pts, dx = kernel_density_support(x, bw)
        fx_hat = kernel_density(x, x_pts, bw)
        emp_risk_1[i] = sum(pow(fx_hat, 2)) * dx

        # Cross-validation to find optimal bandwidth
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            x_train, x_test = x[train_index], x[test_index]
            fx_hat = kernel_density(x_train, x_test, bw)
            emp_risk_2[i] = emp_risk_2[i] + fx_hat

        # Compute empirical risk
        emp_risk[i] = emp_risk_1[i] - 2. / n_obs * emp_risk_2[i]
        i += 1

    # Select bandwidth associated to minimum empirical risk
    bw_opt = bw_vec[emp_risk.argmin()]
    return bw_opt


def compute_sev(x, y, method='kernel', cv_opt_bw=False):
    if method == 'linear':
        return np.corrcoef(x, y)[0, 1]
    elif method == 'spearman':
        return stats.spearmanr(x, y).correlation
    elif method == 'kernel':
        bw_scott = get_bw_scott(x)  # Scott's Rule of Thumb

        if np.isnan(bw_scott):
            return np.nan

        if cv_opt_bw:
            bw = get_optimum_cv_bandwidth(x, bw_scott)
        else:
            bw = bw_scott

        # Compute the support where the density of x is to be estimated
        x_pts, dx = kernel_density_support(x, bw)

        # Compute an estimate of g(x) = E[Y | X = x] via kernel regression
        [g_hat, fx_hat] = nada_wats_estim(x, y, x_pts, bw)

        return float((sum(g_hat ** 2 * fx_hat) * dx - pow(y.mean(), 2)) / y.var(ddof=1))


def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_sev_multiproc(d, x_df, y_df, method='kernel', cv_opt_bw=False):
    # Suppress warning message caused by NaNs in the dataframes for kernel regr.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

        for tkr in x_df.columns:
            sev = compute_sev(x_df[tkr].values, y_df.values, method, cv_opt_bw)
            d[x_df[tkr].name] = sev


class StockPicker:
    def __init__(self,
                 n_stk: int,
                 algo: str,
                 ret_df: pd.Series,
                 vol_df: pd.Series,
                 sharpe_df: pd.Series,
                 mktcap_df: pd.Series = None,
                 nsel: float = 0.1):
        self.n_stk = n_stk
        self.algo = algo
        self.ret_df = ret_df
        self.vol_df = vol_df
        self.sharpe_df = sharpe_df
        self.cap_df = mktcap_df
        self.nsel = nsel

    def use_momentum(self,
                     long_only: bool = True,
                     risk_managed: bool = True) -> pd.DataFrame:
        if risk_managed:
            mom_df = self.sharpe_df.copy()
        else:
            mom_df = self.ret_df.copy()
        mom_df = mom_df.mean(skipna=False)
        mom_df = mom_df.sort_values(ascending=False)
        mom_df = mom_df.dropna()
        if long_only:
            stock_df = mom_df.head(self.n_stk).to_frame()
            stock_df.columns = ['metric']
            stock_df['Pos'] = 1
        else:
            median_mom = np.quantile(mom_df, 0.5)
            tickers = mom_df.subtract(median_mom).abs().head(self.n_stk).index
            stock_df = mom_df.loc[tickers].to_frame()
            stock_df.columns = ['Metric']
            stock_df.loc[stock_df['Metric'] > median_mom, 'Pos'] = 1
            stock_df.loc[stock_df['Metric'] < median_mom, 'Pos'] = -1

        # n_quantiles = 6
        # quantiles = np.quantile(mom_df, np.linspace(0, 1, n_quantiles + 1))
        # q = self.n_stk
        # mom_df = mom_df[(quantiles[q - 1] < mom_df) & (mom_df <= quantiles[q])]
        # if risk_managed:
        #     algo = 'rmmtm'
        # else:
        #     algo = 'mtm'
        # filename = f'./{algo}_q{q}_metric.txt'
        # file_handle = open(filename, 'a')
        # file_handle.write(f'{mom_df.mean():.5f}\n')
        # file_handle.close()
        return stock_df

    def build_high_sr_index(self, risk_managed: bool = True):
        # Create artificial (high Sharpe ratio) index
        hsr_idx = pd.Series(dtype=float)
        for d in self.sharpe_df.index:
            sr_df_curr = self.sharpe_df.copy().loc[d]
            sr_df_curr = sr_df_curr.sort_values(ascending=False)
            sr_df_curr = sr_df_curr.head(round(self.nsel * sr_df_curr.size))
            # sr_df_curr = sr_df_curr.sample(n=round(self.nsel * sr_df_curr.size))
            sr_df_curr = sr_df_curr.fillna(0)

            mkt_cap_curr = pd.Series(index=sr_df_curr.index, dtype=float)
            tickers_with_market_cap = [t for t in self.cap_df.columns if t in sr_df_curr.index]
            mkt_cap_curr.loc[tickers_with_market_cap] = self.cap_df.loc[d, tickers_with_market_cap]

            if any(np.isnan(mkt_cap_curr)):
                # missing_lst = list(mkt_cap_curr[np.isnan(mkt_cap_curr)].index)
                # print(' --- Warning! Market capitalization missing for ' +
                #       " ".join(str(x) for x in missing_lst) + ' --- ')

                # Fill nans with zeros
                mkt_cap_curr = mkt_cap_curr.fillna(0)

            if risk_managed:
                hsr_idx_curr = sum(sr_df_curr * mkt_cap_curr)
            else:
                ret_df_curr = self.ret_df.loc[d, sr_df_curr.index]
                hsr_idx_curr = sum(ret_df_curr * mkt_cap_curr)

            hsr_idx[d] = hsr_idx_curr / sum(mkt_cap_curr)
        return hsr_idx

    def use_sev(self,
                risk_managed: bool = True,
                long_only: bool = True,
                multi_proc: bool = True,
                method: str = 'kernel',
                tracking_mode: bool = False,
                cv_opt_bw: bool = False) -> pd.DataFrame:
        # Build artificial high Sharpe ratio index
        hsr_idx = self.build_high_sr_index(risk_managed)

        # In case of risk-managed algo flavor, correlate Sharpe ratios
        if risk_managed:
            features_df = self.sharpe_df
        else:
            features_df = self.ret_df

        if not tracking_mode:
            features_df = features_df.subtract(hsr_idx, axis=0)

        tickers = self.ret_df.columns

        if multi_proc:
            # Reset shared dictionary of SEV values
            shrd_dct = multiprocessing.Manager().dict()
            shrd_dct.clear()

            jobs = []
            jobs_running = 0
            max_jobs_running = multiprocessing.cpu_count()

            # Compute SEV metric for each stock using multiprocessing
            for tkr_cnk in chunks(tickers, int(len(tickers) / max_jobs_running)):
                args = (shrd_dct, features_df[tkr_cnk], hsr_idx, method, cv_opt_bw)
                p = multiprocessing.Process(target=compute_sev_multiproc,
                                            args=args)
                jobs.append(p)
                p.start()

                jobs_running += 1
                if jobs_running >= max_jobs_running:
                    while jobs_running >= max_jobs_running:
                        jobs_running = 0
                        for p in jobs:
                            jobs_running += p.is_alive()

            for p in jobs:
                p.join()

            # Convert dict to df, remove NaNs, sort by descending SEV value and pick top entries
            sev_df = pd.DataFrame(shrd_dct.values(), index=shrd_dct.keys(), columns=['Metric'])
        else:
            sev_df = pd.DataFrame(np.nan, index=tickers, columns=['Metric'])
            for tkr in tickers:
                sev_df.loc[tkr, 'Metric'] = compute_sev(x=features_df[tkr].values,
                                                        y=hsr_idx.values,
                                                        method=method,
                                                        cv_opt_bw=cv_opt_bw)

        sev_df = sev_df.sort_values(ascending=False, by='Metric')
        sev_df = sev_df.dropna()

        if long_only:
            stock_df = sev_df.head(self.n_stk).copy()
            stock_df['Pos'] = 1
            stock_df = stock_df[stock_df.Metric > 0.2]
        else:
            median_sev = np.quantile(sev_df, 0.5)
            tickers = sev_df.subtract(median_sev).abs().head(self.n_stk).index
            stock_df = sev_df.loc[tickers, :]
            stock_df.loc[stock_df['Metric'] > median_sev, 'Pos'] = 1
            stock_df.loc[stock_df['Metric'] < median_sev, 'Pos'] = -1

        # n_quantiles = 6
        # quantiles = np.quantile(sev_df, np.linspace(0, 1, n_quantiles + 1))
        # q = self.n_stk
        # sev_df = sev_df[(quantiles[q - 1] < sev_df['Metric']) & (sev_df['Metric'] <= quantiles[q])]

        # if risk_managed:
        #     algo = 'rmsev'
        # else:
        #     algo = 'sev'
        # filename = f'./{algo}_q{q}_metric.txt'
        # file_handle = open(filename, 'a')
        # file_handle.write(f"{sev_df['Metric'].mean():.5f}\n")
        # file_handle.close()

        # return sev_df
        return stock_df

    def use_backward_subsel(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="delta_grad == 0.0")
            warnings.filterwarnings("ignore", message="Singular Jacobian matrix")

            # Start by selecting only stocks with no missing data in time series            
            stk_sel = list(set(self.ret_df.dropna(axis=1).columns) & set(self.vol_df.dropna(axis=1).columns))

            while True:
                stk_rnd = random.sample(stk_sel, self.ret_df.shape[0])

                opt_weights = mv_opt.get_opt_weights(self.ret_df[stk_rnd],
                                                     self.vol_df[stk_rnd],
                                                     allow_short_sell=False,
                                                     risk_avers=None,
                                                     max_lvrg=2.0,
                                                     mv_improv=False,
                                                     verbose=False)

                stk_min_w = list(opt_weights[abs(opt_weights) == abs(opt_weights).min()].index)[0]
                stk_sel.remove(stk_min_w)

                if len(stk_sel) <= self.n_stk:
                    break

                print(f'Number of stocks selected: {len(stk_sel)}')
            return pd.DataFrame(np.nan, index=stk_sel, columns=[])

    def pick_stocks(self, multi_proc: bool = True, cv_opt_bw: bool = False) -> pd.DataFrame:
        algo = self.algo.lower()
        if algo == 'mom' or algo == 'mtm':
            stock_df = self.use_momentum(long_only=False, risk_managed=False)
        elif algo == 'lomom' or algo == 'lomtm':
            stock_df = self.use_momentum(long_only=True, risk_managed=False)
        elif algo == 'rmmom' or self.algo == 'rmmtm':
            stock_df = self.use_momentum(long_only=False, risk_managed=True)
        elif algo == 'lormmom' or self.algo == 'lormmtm':
            stock_df = self.use_momentum(long_only=True, risk_managed=True)
        elif algo == 'sev':
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    cv_opt_bw=cv_opt_bw)
        elif algo == 'rmsev':
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    cv_opt_bw=cv_opt_bw)
        elif algo == 'lin':
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=False,
                                    multi_proc=multi_proc,
                                    method='linear',
                                    tracking_mode=True)
        elif algo == 'lolin':
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    method='linear',
                                    tracking_mode=True)
        elif algo == 'rmlin':
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=False,
                                    multi_proc=multi_proc,
                                    method='linear',
                                    tracking_mode=True)
        elif algo == 'lormlin':
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    method='linear',
                                    tracking_mode=True)
        elif algo == 'imv':
            stock_df = self.use_backward_subsel()

        else:
            raise Exception('Algorithm not recognized.')

        return stock_df
