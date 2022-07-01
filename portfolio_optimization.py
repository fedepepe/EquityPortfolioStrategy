#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:37:30 2021

@author: federico
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
import math
from scipy.stats import norm

import proposed_estim


def portfolio_volatility(w, cov_mat):
    return np.sqrt(w.dot(cov_mat).dot(w))


def quadratic_fun(w, mu, cov_mat, phi=None):
    """ w  : vector of weights
        mu : vector of mean returns
        cov_mat : covariance matrix of returns
        phi : risk aversion coefficient """

    if phi is None:
        # return - w.dot(mu) / math.sqrt(abs(w.dot(cov_mat).dot(w)))  # Tangency portfolio
        w_tr = w / w.dot(mu)         # Fast algorithm
        return w_tr.dot(cov_mat).dot(w_tr)
    
    elif math.isinf(phi):
        return w.dot(cov_mat).dot(w)  # Minimum variance portfolio

    else:
        return phi / 2. * w.dot(cov_mat).dot(w) - w.dot(mu)  # Optimized portfolio


def shortfall_objective(w, ret_df, bema_ret_df=None, shortfall=-0.01, level=0.95):
    exc_ret_df = ret_df.values.dot(w.transpose())
    if bema_ret_df is not None:
        exc_ret_df = exc_ret_df - bema_ret_df.values
    mu = np.nanmean(exc_ret_df)
    sigma = np.nanstd(exc_ret_df)
    return norm.ppf(level) * sigma - mu + shortfall  # This must be <= 0


def compute_risk_contributions(w, cov_mat, sigma_pf):
    w = np.matrix(w)
    marg_risk_contrib = cov_mat * w.T     # marginal risk contribution
    risk_contrib = np.multiply(marg_risk_contrib, w.T) / sigma_pf      # risk contribution
    return risk_contrib


def risk_budget_objective(w, risk_w_target, cov_mat):
    sigma_pf = portfolio_volatility(w, cov_mat)
    risk_target = risk_w_target * sigma_pf
    asset_rc = compute_risk_contributions(w=w, cov_mat=cov_mat, sigma_pf=sigma_pf)
    return sum(np.square(asset_rc - risk_target.T))[0, 0]


def leverage(w):
    return abs(w).sum()


def get_opt_weights(ret_df: pd.DataFrame,
                    vol_df: pd.DataFrame,
                    method: str = 'meanvar',
                    allow_short_sell: bool = True,
                    risk_avers: float = None,
                    max_lvrg: float = 3.0,
                    bema_ret_df: pd.Series = None,
                    mv_improv: bool = True,
                    tol: float = 1e-8,
                    verbose: bool = False):

    ret_df_copy = ret_df.copy()
    vol_df_copy = vol_df.copy()
    
    # clean data
    ret_df_copy = ret_df_copy.dropna(axis=1)
    ret_df_copy = ret_df_copy.loc[:, (ret_df_copy != ret_df_copy.iloc[0]).any()]

    if (method == 'meanvar') and mv_improv:
        vol_df_copy = vol_df_copy.dropna(axis=1)
        ticker_list = [t for t in ret_df_copy.columns if t in vol_df_copy.columns]
        ret_df_copy = ret_df_copy[ticker_list]
        vol_df_copy = vol_df_copy[ticker_list]
        
    n_stk = len(ret_df_copy.columns)
    # ones = np.ones(n_stk)
    ones = pd.Series(1, index=ret_df_copy.columns)

    if n_stk == 0:
        return pd.Series(dtype=float)

    cons_sum = opt.LinearConstraint(ones, 1, 1)     # Imposing sum of weights = 1
    cons_lev = opt.NonlinearConstraint(leverage, 0, max_lvrg)   # Imposing maximum leverage
    constraints = (cons_sum, cons_lev)
    
    max_abs_wght = .5     # Maximum single portfolio weight (to promote diversification)
    
    # Imposing the no short-selling restriction
    if allow_short_sell:
        bounds = opt.Bounds(- max_abs_wght * ones, max_abs_wght * ones)
    else:
        bounds = opt.Bounds(0 * ones, max_abs_wght * ones)

    # Initial starting point is the equally-weighted portfolio
    w_eq = 1. / n_stk * ones

    try:
        if method == 'meanvar':
            # Compute vector of mean returns and covariance matrix
            if not mv_improv:
                mu = ret_df_copy.mean().values  # vector of mean returns
                cov_mat = ret_df_copy.cov().values  # covariance matrix
            else:
                mu, cov_mat = proposed_estim.proposed_estim(ret_df_copy, vol_df_copy)

            fun = quadratic_fun
            args = (mu, cov_mat, risk_avers)

        elif method == 'erc':
            cov_mat = ret_df_copy.cov().values
            fun = risk_budget_objective
            args = (w_eq.values, cov_mat)

        elif method == 'shortfall':
            fun = shortfall_objective
            args = (ret_df_copy, bema_ret_df)
        else:
            raise Exception('Optimization method not recognized.')

        # Loop until optimization is successful
        w0 = w_eq
        success = False
        maxiter = 100
        while not success:
            opt_result = opt.minimize(fun, w0, args=args, method='trust-constr',
                                      options={'verbose': verbose,
                                               'xtol': tol, 'gtol': tol, 'barrier_tol': tol,
                                               'maxiter': maxiter},
                                      constraints=constraints, bounds=bounds)
            success = opt_result.success

            w0 = np.random.uniform(low=0.0, high=max_abs_wght, size=n_stk) * ones
            w0 = w0 / sum(w0)
            maxiter += 100
            
        if method == 'shortfall' and opt_result.fun > 0:
            message = ("\n --- No portfolio found satisfying shortfall constraint. "
                       f"Decrease shortfall by {100 * opt_result.fun:.2f}% --- ")
            print(message)
            
        return pd.Series(opt_result.x, index=ret_df_copy.columns)
        
    except Exception:
        message = (" --- Mean-variance optimization failed! "
                   "Using equally-weighted portfolio --- ")
        print(message)
        # breakpoint()
        return w_eq
