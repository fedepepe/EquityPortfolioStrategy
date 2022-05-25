#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import pandas as pd
import numpy as np
import math
import scipy.optimize as opt
import risk_models
import matplotlib.pyplot as plt
import proposed_estim
import label_lines
import csv

from yahoo_data_downloader import YahooDataDownloader


def quadratic_util_fun(w, mu, cov_mat, phi):
    # w  : vector of weights
    # mu : vector of mean returns
    # cov_mat : covariance matrix of returns
    # phi : risk aversion coefficient
    
    if phi is None:
        # Compute risk aversion factor yielding the tangency portfolio
        # Sigma_np = cov_mat.to_numpy()
        # invSigma_np = np.linalg.inv(Sigma_np)
        # invSigma = pd.DataFrame(data = invSigma_np, index = cov_mat.index, columns = cov_mat.columns)
        # series_of_ones = pd.Series(1, index=mu.index)
        # phi0 = series_of_ones.dot(invSigma).dot(mu)
        # return phi0/2 * w.dot(cov_mat).dot(w) - w.dot(mu)  # Tangency portfolio
        # Alternative single-line (equivalent) code:
        return - w.dot(mu) / math.sqrt(abs(w.dot(cov_mat).dot(w)))  # Tangency portfolio
        # abs just prevents the run from failing due to numerical errors (causing w*Sigma*w being < 0)
    elif math.isinf(phi):
        return w.dot(cov_mat).dot(w)  # Minimum variance portfolio
    else:
        return phi / 2. * w.dot(cov_mat).dot(w) - w.dot(mu)  # Optimized portfolio

def leverage(w):
    return abs(w).sum()

def plot_portfolio(mu_p, sig2_p, sym_str='bo', figure=None):
    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
    else:
        ax = figure.axes[0]
    ax.plot(100. * np.sqrt(252 * sig2_p), 100. * 252 * mu_p, sym_str, markersize=3, 
            markeredgewidth=0, alpha=0.5)
    return figure


# Load dataset of daily returns and volatilities
dataset = 'SP500'  # SP500 or STOXXE600

YahooDataDownloaderObj = YahooDataDownloader(dataset)

last_price_df, return_df, real_vol_df, mktcap_df, _ = YahooDataDownloaderObj.get_stock_data()

# Remove columns with at least one NaN
return_df = return_df[return_df.columns.intersection(mktcap_df.columns)].dropna(axis=1)
real_vol_df = real_vol_df[real_vol_df.columns.intersection(mktcap_df.columns)].dropna(axis=1)

mkt_ret_df, mkt_idx_df = YahooDataDownloaderObj.get_mkt_data(return_df.index)

# %% Set simulation parameters
allow_short_sell = False
risk_avers =  None     # Risk aversion coefficient for quadratic utility function
max_lvrg = 1.           # Maximum leverage (i.e., L1 norm of portfolio weights vector)
max_weight = .5         # Maximum single portfolio weight (to promote diversification)

n_stck = 100             # Size of investment universe
n_run = 250             # Number of simulation runs

data_filename = f'./results/pf_mu_sig_stk{n_stck}_rac{risk_avers}_lvg{max_lvrg}.txt'

# %% Do computations
n_done = 0
while n_done < n_run:
    # Randomly pick n_stck stocks out of the SP500 components to form the investment universe
    idx_stk_sel = np.random.choice(len(return_df.columns), size=n_stck, replace=False)

    ret_df = return_df.iloc[:,idx_stk_sel]
    vol_df = real_vol_df.iloc[:,idx_stk_sel]

    stock_idx = ret_df.columns

    ret_df_train, ret_df_test = np.split(ret_df, [int(.75*len(ret_df))])
    vol_df_train, vol_df_test = np.split(vol_df, [int(.75*len(vol_df))])
    mkt_ret_df_train, mkt_ret_df_test = np.split(mkt_ret_df, [int(.75*len(mkt_ret_df))])
    
    cov_shr = risk_models.CovarianceShrinkage(prices=ret_df_train, returns_data=True)
    
    series_of_ones = pd.Series(1, index=stock_idx)

    # Imposing sum of weights being equal to 1
    linear_constraint = opt.LinearConstraint(series_of_ones, 1, 1)

    # Imposing the restricion on maximum leverage
    nonlinear_constraint = opt.NonlinearConstraint(leverage, 1, max_lvrg)

    # Imposing the no short-selling restriction
    if allow_short_sell:
        bounds = opt.Bounds(-max_weight * series_of_ones, max_weight * series_of_ones)
    else:
        bounds = opt.Bounds(0 * series_of_ones, max_weight * series_of_ones)

    # Equally-weighted portfolio
    w_eq = 1. / n_stck * series_of_ones
    # w0 = w_eq
    
    # Initial starting point is either the equally-weighted portfolio or a random one
    w0 = np.random.uniform(low=0.0, high=1.0, size=n_stck) * series_of_ones
    w0 = w0 / sum(w0)
    
    try:
        ### Optimize portfolio variance over the training set
        # 1. Sigma is the sample covmat of returns
        mu = ret_df_train.mean().values      # vector of mean returns
        cov_mat = ret_df_train.cov().values  # covariance matrix
    
        opt_result = opt.minimize(quadratic_util_fun, w0, args=(mu, cov_mat, risk_avers),
                                  method='trust-constr',
                                  options={'verbose': 3, 'maxiter': 1000},
                                  constraints=(linear_constraint, nonlinear_constraint),
                                  bounds=bounds)
        if not opt_result.success:
            continue
        
        # 2. Sigma is given by the proposed estimator
        mu_impr, cov_mat_impr = proposed_estim.proposed_estim(ret_df_train, vol_df_train)
    
        opt_result_impr = opt.minimize(quadratic_util_fun, w0, args=(mu_impr, cov_mat_impr, risk_avers),
                                  method='trust-constr',
                                  options={'verbose': 3, 'maxiter': 1000},
                                  constraints=(linear_constraint, nonlinear_constraint),
                                  bounds=bounds)

        if not opt_result_impr.success:
            continue
    
        # 3. Sigma is given by Ledoid-Wolf shrinkage towards identity matrix
        cov_mat_lwcv = cov_shr.ledoit_wolf(shrinkage_target="constant_variance")
    
        opt_result_lwcv = opt.minimize(quadratic_util_fun, w0, args=(mu, cov_mat_lwcv, risk_avers),
                                  method='trust-constr',
                                  options={'verbose': 3, 'maxiter': 1000},
                                  constraints=(linear_constraint, nonlinear_constraint),
                                  bounds=bounds)
        
        if not opt_result_lwcv.success:
            continue
        
        # 4. Sigma is given by Ledoid-Wolf shrinkage towards single factor
        cov_mat_lw1f = cov_shr.ledoit_wolf(shrinkage_target="single_factor", xmkt=mkt_ret_df_train)
    
        opt_result_lw1f = opt.minimize(quadratic_util_fun, w0, args=(mu, cov_mat_lw1f, risk_avers),
                                  method='trust-constr',
                                  options={'verbose': 3, 'maxiter': 1000},
                                  constraints=(linear_constraint, nonlinear_constraint),
                                  bounds=bounds)
        
        if not opt_result_lw1f.success:
            continue
        
    except np.linalg.LinAlgError:
        continue
    
    ### Compute out-of-sample mean and variance
    # Using sample covariance matrix of returns
    w_opt = pd.Series(opt_result.x, index=stock_idx)
    pf_ret = ret_df_test.values.dot(w_opt)
    mu_p = pf_ret.mean()
    sig2_p = pf_ret.var()

    # Using proposed estimator of covariance matrix
    w_opt_impr = pd.Series(opt_result_impr.x, index=stock_idx)
    pf_ret = ret_df_test.values.dot(w_opt_impr)
    mu_p_impr = pf_ret.mean()
    sig2_p_impr = pf_ret.var()

    # Using Ledoid-Wolf shrinkage towards identity matrix
    w_opt_lwcv = pd.Series(opt_result_lwcv.x, index=stock_idx)
    pf_ret = ret_df_test.values.dot(w_opt_lwcv)
    mu_p_lwcv = pf_ret.mean()
    sig2_p_lwcv = pf_ret.var()

    # Using single-factor Ledoid-Wolf shrinkage
    w_opt_lw1f = pd.Series(opt_result_lw1f.x, index=stock_idx)
    pf_ret = ret_df_test.values.dot(w_opt_lw1f)
    mu_p_lw1f = pf_ret.mean()
    sig2_p_lw1f = pf_ret.var()
    
    # Using identity matrix (i.e., equally-weighted portfolio)
    w_opt_ew = w_eq
    pf_ret = ret_df_test.values.dot(w_opt_ew)
    mu_p_ew = pf_ret.mean()
    sig2_p_ew = pf_ret.var()

    # Print data to text file
    new_data = np.array([[mu_p, sig2_p, mu_p_impr, sig2_p_impr, mu_p_lwcv, sig2_p_lwcv, 
                          mu_p_lw1f, sig2_p_lw1f, mu_p_ew, sig2_p_ew]])
    
    with open(data_filename, "a") as f:
        np.savetxt(f, new_data, fmt='%.8f')

    n_done += 1

    print(f'{n_done}', end='..')

#%% Load data and show the scatter plot of simulated portfolios
with open(data_filename, newline='') as results_handle:
    results_reader = csv.reader(results_handle, delimiter=' ')
    col_names = ['mu_p', 'sig2_p', 'mu_p_impr', 'sig2_p_impr', 'mu_p_lwcv', 'sig2_p_lwcv', 
                 'mu_p_lw1f', 'sig2_p_lw1f', 'mu_p_ew', 'sig2_p_ew']
    results_df = pd.DataFrame(data=results_reader, columns=col_names)

# Convert columns to numerics
results_df = results_df.apply(pd.to_numeric)

figure = None        
figure = plot_portfolio(results_df['mu_p'], results_df['sig2_p'], 'ob', figure)
figure = plot_portfolio(results_df['mu_p_impr'], results_df['sig2_p_impr'], 'or', figure)
figure = plot_portfolio(results_df['mu_p_lwcv'], results_df['sig2_p_lwcv'], 'og', figure)
figure = plot_portfolio(results_df['mu_p_lw1f'], results_df['sig2_p_lw1f'], 'om', figure)
figure = plot_portfolio(results_df['mu_p_ew'], results_df['sig2_p_ew'], 'oy', figure)

ax = figure.axes[0]
ax.set_xlabel('Standard deviation [%]', fontsize=12)
ax.set_ylabel('Mean return [%]', fontsize=12)
legend = ax.legend(['Sample cov. matrix', 'Proposed estim.', 'Ledoit-Wolf (const. var.)',
           'Ledoit-Wolf (mkt. factor)', 'Equally weighted'], edgecolor="black")
legend.get_frame().set_alpha(1)
legend.get_frame().set_facecolor((1, 1, 1, 1))

# Plot contour lines with equal Sharpe ratio        
xlim = np.asarray(ax.get_xlim())
plt.autoscale(False)    #Turn autoscaling off
SR = [0.25, 0.5, 0.75, 1., 1.25, 1.5]
for sr in SR:
    ax.plot(xlim, sr*xlim, '--k', linewidth=0.5, label='SR='+str(sr))

label_lines.label_lines(ax.get_lines()[-len(SR):], x_vals=1.03 * xlim[0] * np.ones((len(SR), 1)),
                        zorder=2.5, fontsize=8)

img_filename = f'./results/pf_mu_sig_stk{n_stck}_rac{risk_avers}_lvg{max_lvrg}.pdf'
figure.savefig(img_filename, format='pdf', bbox_inches='tight')