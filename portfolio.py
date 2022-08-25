#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:03:44 2021

@author: federico
"""
import logging

import pandas as pd
import numpy as np
import portfolio_optimization as mv_opt
import portfolio_analysis_tools as pf_analysis
from portfolio_metrics import EnumPerfMetrics, PerfMetric
import warnings
from typing import Union


class Portfolio:
    def __init__(self, endow: float = 1e5):
        self.holdings_curr = {'cash': endow}  # Current n. of stocks hold
        self.holdings_hist = pd.DataFrame()  # Historical n. of stocks hold
        self.val_stk_curr = {'cash': endow}  # Current value of stocks hold
        self.val_stk_hist = pd.DataFrame()  # Historical value of stocks hold
        self.val_tot_curr = endow  # Current overall portfolio value
        self.val_tot_hist = pd.Series(dtype=float)  # Historical overall portfolio value
        self.val_trans_curr = 0.  # Current amount of wealth traded
        self.val_trans_hist = pd.Series(dtype=float)  # Historical amount of wealth traded
        self.perf_metrics = {EnumPerfMetrics.RETS: PerfMetric(label='returns'),
                             EnumPerfMetrics.LOG_RETS: PerfMetric(label='log_returns'),
                             EnumPerfMetrics.EXC_RETS: PerfMetric(label='exc_returns'),
                             EnumPerfMetrics.ANN_MEAN_RET: PerfMetric(label='return', format='+.2%'),
                             EnumPerfMetrics.ANN_MEAN_EXC_RET: PerfMetric(label='xret', format='+.2%'),
                             EnumPerfMetrics.VOLATILITY: PerfMetric(label='volat', format='.2%', rev_color_scale=True),
                             EnumPerfMetrics.SHARPE: PerfMetric(label='sharpe'),
                             EnumPerfMetrics.IC: PerfMetric(label='i.c.'),
                             EnumPerfMetrics.SORTINO: PerfMetric(label='sortino'),
                             EnumPerfMetrics.STAR: PerfMetric(label='starr', format='.4f'),
                             EnumPerfMetrics.TURNOVER: PerfMetric(label='trnovr', format='.2%', rev_color_scale=True),
                             EnumPerfMetrics.ALPHA: PerfMetric(label='alpha', format='+.2%'),
                             EnumPerfMetrics.BETA: PerfMetric(label='beta', rev_color_scale=True),
                             EnumPerfMetrics.PVAL: PerfMetric(label='pval', rev_color_scale=True),
                             EnumPerfMetrics.DRAWDOWN: PerfMetric(label='drawdown'),
                             EnumPerfMetrics.MAXDD: PerfMetric(label='maxdd', format='.2%', rev_color_scale=True),
                             EnumPerfMetrics.FF_FACTORS: PerfMetric(label='ff_factors')}

    def trade_stock(self, tckr, price, quantity, trsctn_fee_fix, trsctn_fee_prop):
        # Compute the transaction value
        trsctn_value = quantity * price

        # Keep trace of the wealth traded
        self.val_trans_curr = self.val_trans_curr + abs(trsctn_value)

        # Exchange cash with stock
        self.holdings_curr['cash'] = self.holdings_curr['cash'] - trsctn_value
        self.val_stk_curr['cash'] = self.val_stk_curr['cash'] - trsctn_value
        # Subtract transaction costs
        trsctn_cost = trsctn_fee_fix + trsctn_fee_prop * abs(trsctn_value)
        self.holdings_curr['cash'] = self.holdings_curr['cash'] - trsctn_cost
        self.val_stk_curr['cash'] = self.val_stk_curr['cash'] - trsctn_cost
        # Update the number of stocks hold
        if tckr in self.holdings_curr:
            self.holdings_curr[tckr] = self.holdings_curr[tckr] + quantity
            self.val_stk_curr[tckr] = self.val_stk_curr[tckr] + trsctn_value
        else:
            self.holdings_curr[tckr] = quantity
            self.val_stk_curr[tckr] = trsctn_value

        # If there are no stocks left, remove the entry from the portfolio dictionary
        if self.holdings_curr[tckr] == 0:
            del self.holdings_curr[tckr]
            del self.val_stk_curr[tckr]

        # Update the current portfolio value (change is only due to transaction costs)
        self.val_tot_curr = self.val_tot_curr - trsctn_cost

    def rebalance(self, stock_df_new, price_dct, trsctn_fee_fix, trsctn_fee_prop):
        tckrs = stock_df_new.index.tolist()

        # First, liquidate open positions that are no longer needed
        tckrs_to_sell = [tckr for tckr in list(self.holdings_curr.keys()) if tckr not in tckrs]
        tckrs_to_sell.remove('cash')
        for tckr in tckrs_to_sell:
            quantity = -self.holdings_curr[tckr]
            if not np.isnan(price_dct[tckr]):
                self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

        # Second, update quantity of stocks already present in the portfolio
        tckrs_to_update = [tckr for tckr in list(self.holdings_curr.keys()) if tckr in tckrs]
        for tckr in tckrs_to_update:
            quantity = stock_df_new.loc[tckr, 'Quantity'] - self.holdings_curr[tckr]
            if not np.isnan(price_dct[tckr]):
                self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

        # Third, buy new stocks
        tckrs_to_buy = [tckr for tckr in tckrs if tckr not in list(self.holdings_curr.keys())]
        for tckr in tckrs_to_buy:
            quantity = stock_df_new.loc[tckr, 'Quantity']
            self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

    def update_value(self, price_dct):
        total_value = .0
        for tckr in self.holdings_curr:
            if tckr in price_dct:
                if not np.isnan(price_dct[tckr]):
                    stk_value = self.holdings_curr[tckr] * price_dct[tckr]
                    self.val_stk_curr[tckr] = stk_value
                    total_value = total_value + stk_value
                else:
                    print(f'Missing price for {tckr}!')
            elif tckr == 'cash':
                total_value = total_value + self.holdings_curr['cash']

        self.val_tot_curr = total_value
        self.val_trans_curr = 0.

    def update_hist(self, timestamp, save_stk_hist=True):
        if save_stk_hist:
            holdings_curr_df = pd.DataFrame(data=[list(self.holdings_curr.values())],
                                            index=[timestamp],
                                            columns=list(self.holdings_curr.keys()))
            self.holdings_hist = pd.concat([self.holdings_hist, holdings_curr_df])

            val_stk_curr_df = pd.DataFrame(data=[list(self.val_stk_curr.values())],
                                           index=[timestamp],
                                           columns=list(self.val_stk_curr.keys()))
            self.val_stk_hist = pd.concat([self.val_stk_hist, val_stk_curr_df])

        self.val_tot_hist = pd.concat([self.val_tot_hist,
                                       pd.Series(self.val_tot_curr, index=[timestamp])])
        self.val_trans_hist = pd.concat([self.val_trans_hist,
                                         pd.Series(self.val_trans_curr, index=[timestamp])])

    def compute_weights(self, stock_sel_df, wght_mtd, risk_avers, max_lvrg,
                        mktcap_df=None, ret_df=None, vol_df=None, bema_ret_df=None,
                        trsctn_fee_fix=0, trsctn_fee_prop=0):

        stock_df = stock_sel_df.copy()

        # Add columns with market capitalization or volatility, if needed
        if wght_mtd == 'mktcap':
            mktcap_df_copy = mktcap_df.reindex(index=stock_df.index)
            if any(np.isnan(mktcap_df_copy)):
                mktcap_df_copy = mktcap_df_copy.fillna(0)
            stock_df['MktCap'] = mktcap_df_copy[stock_df.index]
        elif wght_mtd == 'riskpar':
            stock_df['Volat'] = np.sqrt(pow(vol_df[stock_df.index], 2).sum(skipna=False))

        # Discard stocks for which I don't have a price (or, in case, a market cap. or volatility)
        if stock_df.isnull().values.any():
            stock_df = stock_df.dropna().copy()
            ret_df = ret_df[stock_df.index]
            vol_df = vol_df[stock_df.index]

        # Final list of selected stocks
        tckrs_sel = stock_df.index

        # Compute the number of stocks to be traded by first computing the amount of wealth
        # to be allocated and then dividing by the price
        if wght_mtd.lower() == 'equal':  # Equally-weighted portfolio
            stock_df['Weight'] = [1 / tckrs_sel.size for _ in tckrs_sel]
            stock_df['Weight'] = stock_df['Pos'] * stock_df['Weight']
        elif wght_mtd.lower() == 'metric':  # Metric (momentum or SEV)-weighted portfolio
            stock_df['Weight'] = stock_df['metric'] / stock_df['metric'].abs().sum()
            stock_df['Weight'] = stock_df['Pos'] * stock_df['Weight']
        elif wght_mtd.lower() == 'erc':  # Equally risk contribution portfolio
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="delta_grad == 0.0")
                warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
                opt_weights = mv_opt.get_opt_weights(ret_df=ret_df,
                                                     vol_df=vol_df,
                                                     method='erc',
                                                     allow_short_sell=False,
                                                     risk_avers=risk_avers,
                                                     max_lvrg=max_lvrg,
                                                     bema_ret_df=bema_ret_df,
                                                     mv_improv=False)
                stock_df['Weight'] = opt_weights
        elif wght_mtd.lower() == 'riskpar':  # Risk-parity weighted portfolio
            stock_df['Weight'] = pow(stock_df['Volat'], -1) / pow(stock_df['Volat'], -1).sum()
            stock_df['Weight'] = stock_df['Pos'] * stock_df['Weight']
        elif wght_mtd.lower() == 'mktcap':  # Metric (momentum or SEV)-weighted portfolio
            stock_df['Weight'] = stock_df['MktCap'] / stock_df['MktCap'].sum()
            stock_df['Weight'] = stock_df['Pos'] * stock_df['Weight']
        elif wght_mtd.lower() in ['lotp', 'tp', 'ilotp']:  # Optimized portfolio
            allow_short_sell = (wght_mtd.lower() == 'tp')
            mv_improv = (wght_mtd.lower() == 'ilotp')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="delta_grad == 0.0")
                warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
                opt_weights = mv_opt.get_opt_weights(ret_df=ret_df,
                                                     vol_df=vol_df,
                                                     method='meanvar',
                                                     allow_short_sell=allow_short_sell,
                                                     risk_avers=risk_avers,
                                                     max_lvrg=max_lvrg,
                                                     bema_ret_df=bema_ret_df,
                                                     mv_improv=mv_improv)
                stock_df['Weight'] = opt_weights

        if stock_df.isnull().values.any():
            missing_stocks = ", ".join(stock_df[stock_df.isnull().any(axis=1)].index.to_list())
            logging.warning(f'Could not allocate {missing_stocks} with {wght_mtd}')
            stock_df['Weight'] = stock_df['Weight'].fillna(value=0)

        # Wealth to be allocated in the different assets
        stock_df['Quantity'] = self.val_tot_curr * stock_df['Weight']

        # Divide by the price (also accounting for transaction costs)
        stock_df['Quantity'] /= (trsctn_fee_fix + (1. + trsctn_fee_prop) * stock_df['Price'])

        # Apply floor() function to get an integer number of stocks
        # stock_df['Quantity'] = np.floor(stock_df['Quantity'])

        return stock_df

    def compute_returns_volat_sharpe(self, risk_free_ret: Union[float, pd.Series] = 0):  # Discrete returns
        ret_vol_sharpe = pf_analysis.compute_returns_volat_sharpe(self.val_tot_hist, risk_free_ret)
        self.perf_metrics[EnumPerfMetrics.RETS].value = ret_vol_sharpe[0]
        self.perf_metrics[EnumPerfMetrics.LOG_RETS].value = ret_vol_sharpe[1]
        self.perf_metrics[EnumPerfMetrics.EXC_RETS].value = ret_vol_sharpe[2]
        self.perf_metrics[EnumPerfMetrics.ANN_MEAN_RET].value = ret_vol_sharpe[3]
        self.perf_metrics[EnumPerfMetrics.ANN_MEAN_EXC_RET].value = ret_vol_sharpe[4]
        self.perf_metrics[EnumPerfMetrics.VOLATILITY].value = ret_vol_sharpe[5]
        self.perf_metrics[EnumPerfMetrics.SHARPE].value = ret_vol_sharpe[6]

    def compute_info_coeff(self, mkt_ret_df: pd.Series):
        ic = pf_analysis.compute_info_ratio(self.val_tot_hist, mkt_ret_df)
        self.perf_metrics[EnumPerfMetrics.IC].value = ic

    def compute_sortino(self,
                        risk_free_ret: Union[float, pd.Series] = 0,
                        min_acc_ret: float = 0):  # Sortino ratio
        sortino = pf_analysis.compute_sortino(self.val_tot_hist, risk_free_ret, min_acc_ret)
        self.perf_metrics[EnumPerfMetrics.SORTINO].value = sortino

    def compute_star(self,
                     risk_free_ret: Union[float, pd.Series] = 0,
                     level: float = 0.05):  # STAR ratio
        star = pf_analysis.compute_star(self.val_tot_hist, risk_free_ret, level)
        self.perf_metrics[EnumPerfMetrics.STAR].value = star

    def compute_turnover(self):  # turnover
        turnover = pf_analysis.compute_turnover(self)
        self.perf_metrics[EnumPerfMetrics.TURNOVER].value = turnover

    def compute_alpha_beta(self,
                           bema_ret_df: pd.Series,
                           risk_free_ret: Union[float, pd.Series] = 0):  # alpha and beta factors
        alpha_beta = pf_analysis.compute_alpha_beta(self.val_tot_hist, bema_ret_df, risk_free_ret)
        self.perf_metrics[EnumPerfMetrics.ALPHA].value = alpha_beta[0]
        self.perf_metrics[EnumPerfMetrics.BETA].value = alpha_beta[1]
        self.perf_metrics[EnumPerfMetrics.PVAL].value = alpha_beta[2][0]

    def compute_drawdown(self):  # drawdown
        drawdown = pf_analysis.compute_drawdown(self.val_tot_hist)
        self.perf_metrics[EnumPerfMetrics.DRAWDOWN].value = drawdown
        self.perf_metrics[EnumPerfMetrics.MAXDD].value = -min(drawdown)

    def compute_ff_factors(self, n_factors):
        ff_factors = pf_analysis.compute_ff_factors(self.val_tot_hist, n_factors)
        self.perf_metrics[EnumPerfMetrics.FF_FACTORS].value = ff_factors

    def analyze(self,
                mkt_ret_df: pd.Series = None,
                risk_free_ret: Union[float, pd.Series] = 0,
                min_acc_ret: float = 0,
                level: float = 0.01):
        self.compute_returns_volat_sharpe(risk_free_ret=risk_free_ret)
        if mkt_ret_df is not None:
            self.compute_info_coeff(mkt_ret_df=mkt_ret_df)
        self.compute_sortino(risk_free_ret=risk_free_ret, min_acc_ret=min_acc_ret)
        self.compute_star(risk_free_ret, level=level)
        self.compute_turnover()
        if mkt_ret_df is not None:
            self.compute_alpha_beta(bema_ret_df=mkt_ret_df, risk_free_ret=risk_free_ret)
        self.compute_drawdown()
