#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:38:09 2021

@author: federico
"""


def proposed_estim(ret_df, volat_df):
    sharpe_df = ret_df / volat_df
    mu_impr = sharpe_df.mean().values  # vector of mean returns
    cov_mat_impr = sharpe_df.cov().values  # covariance matrix
    return mu_impr, cov_mat_impr
