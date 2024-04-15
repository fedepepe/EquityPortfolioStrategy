#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""
# from multiprocessing import managers as mpm

import random
# import statsmodels.api as sm
import warnings

import numpy as np
import pandas as pd
import scipy.optimize as opt

import portfolio_optimization as mv_opt
from compute_sev import compute_sev, compute_sev_multiproc
from definitions import Algorithms, CorrelationMethods


class StockPicker:
    def __init__(self,
                 n_stk: int,
                 algo: Algorithms,
                 ret_df: pd.DataFrame,
                 vol_df: pd.DataFrame,
                 sharpe_df: pd.DataFrame,
                 mktcap_df: pd.DataFrame = None,
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
            try:
                mkt_cap_curr.loc[tickers_with_market_cap] = self.cap_df.loc[d, tickers_with_market_cap]
            except:
                breakpoint()

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
                corr_method: CorrelationMethods = CorrelationMethods.SEV,
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

        if multi_proc:
            sev_df = compute_sev_multiproc(features_df=features_df,
                                           hsr_idx=hsr_idx,
                                           method=corr_method,
                                           cv_opt_bw=cv_opt_bw)
            sev_df.columns = ['Metric']
        else:
            tickers = self.ret_df.columns
            sev_df = pd.DataFrame(np.nan, index=tickers, columns=['Metric'])
            for tkr in tickers:
                sev_df.loc[tkr, 'Metric'] = compute_sev(x=features_df[tkr].values,
                                                        y=hsr_idx.values,
                                                        method=corr_method,
                                                        cv_opt_bw=cv_opt_bw)

        sev_df = sev_df.sort_values('Metric', ascending=False)
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

    def optimize_portfolios(self,
                            long_only: bool = True):
        n_run = 1e3
        max_weight = 0.5
        max_lvrg = 1.5
        n_done = 0
        best_sharpe_curr = - np.inf
        while n_done < n_run:
            # Randomly pick n_stck stocks out of the whole universe
            idx_stk_sel = np.random.choice(len(self.ret_df.columns), size=self.n_stk, replace=False)

            ret_df = self.ret_df.iloc[:, idx_stk_sel]

            tickers_sel = ret_df.columns

            series_of_ones = pd.Series(1, index=tickers_sel)

            # Imposing sum of weights being equal to 1
            linear_constraint = opt.LinearConstraint(series_of_ones, 1, 1)

            # Imposing the restriction on maximum leverage
            nonlinear_constraint_1 = opt.NonlinearConstraint(leverage, 1, max_lvrg)

            # Imposing the no short-selling restriction
            if long_only:
                bounds = opt.Bounds(0 * series_of_ones, max_weight * series_of_ones)
            else:
                bounds = opt.Bounds(- max_weight * series_of_ones, max_weight * series_of_ones)

            # Initial starting point is either the equally-weighted portfolio or a random one
            # w0 = w_eq
            w0 = np.random.uniform(low=0.0, high=1.0, size=self.n_stk) * series_of_ones
            w0 = w0 / sum(w0)

            """Optimize portfolio variance over the training set"""
            # Maximize historical ex-post Sharpe ratio
            try:
                opt_result_hist = opt.minimize(neg_sharpe_ratio, w0,
                                               args=(ret_df),
                                               method='trust-constr',
                                               options={'verbose': False, 'maxiter': 2500},
                                               constraints=(linear_constraint, nonlinear_constraint_1,),
                                               bounds=bounds)
            except (np.linalg.LinAlgError, ValueError):
                continue

            if opt_result_hist.success:
                if opt_result_hist.fun > best_sharpe_curr:
                    best_sharpe_curr = opt_result_hist.fun
                    stock_df = pd.DataFrame(data=opt_result_hist.x, index=tickers_sel, columns=['Metric'])
                    stock_df['Pos'] = 1
        return stock_df

    def pick_stocks(self, multi_proc: bool = True, cv_opt_bw: bool = False) -> pd.DataFrame:
        if self.algo == Algorithms.MTM:
            stock_df = self.use_momentum(long_only=False, risk_managed=False)
        elif self.algo == Algorithms.LOMTM:
            stock_df = self.use_momentum(long_only=True, risk_managed=False)
        elif self.algo == Algorithms.RMMTM:
            stock_df = self.use_momentum(long_only=False, risk_managed=True)
        elif self.algo == Algorithms.LORMMTM:
            stock_df = self.use_momentum(long_only=True, risk_managed=True)
        elif self.algo == Algorithms.SEV:
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    cv_opt_bw=cv_opt_bw)
        elif self.algo == Algorithms.RMSEV:
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    cv_opt_bw=cv_opt_bw)
        elif self.algo == Algorithms.LIN:
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=False,
                                    multi_proc=multi_proc,
                                    corr_method=CorrelationMethods.LINEAR,
                                    tracking_mode=False)
        elif self.algo == Algorithms.LOLIN:
            stock_df = self.use_sev(risk_managed=False,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    corr_method=CorrelationMethods.LINEAR,
                                    tracking_mode=False)
        elif self.algo == Algorithms.RMLIN:
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=False,
                                    multi_proc=multi_proc,
                                    corr_method=CorrelationMethods.LINEAR,
                                    tracking_mode=False)
        elif self.algo == Algorithms.LORMLIN:
            stock_df = self.use_sev(risk_managed=True,
                                    long_only=True,
                                    multi_proc=multi_proc,
                                    corr_method=CorrelationMethods.LINEAR,
                                    tracking_mode=False)
        elif self.algo == Algorithms.MAX_SHARPE:
            stock_df = self.optimize_portfolios(long_only=True)

        # elif self.algo == Algorithms.IMV:
        #     stock_df = self.use_backward_subsel()

        else:
            raise Exception('Algorithm not recognized.')

        return stock_df


def portfolio_returns(return_df, weights):
    cum_ret_df = return_df.add(1).cumprod()
    cum_pf_ret = cum_ret_df.sub(1).values.dot(weights) + 1.
    # pf_ret = np.diff(np.log(cum_pf_ret))
    pf_ret = np.diff(cum_pf_ret) / cum_pf_ret[:-1]
    return pf_ret


def portfolio_sharpe(return_df, weights):
    pf_ret = portfolio_returns(return_df, weights)
    pf_sharpe = pf_ret.mean() / pf_ret.std()
    return pf_sharpe


def leverage(weights):
    return abs(weights).sum()


def neg_sharpe_ratio(weights, return_df):
    pf_sharpe = portfolio_sharpe(return_df, weights)
    return - pf_sharpe


if __name__ == "__main__":
    pass
