#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:59:38 2022

@author: federico
"""

import pandas as pd

dataset = 'STOXXE600'

stk_data = pd.read_pickle('./' + dataset + '/stk_data.pkl')

df = stk_data[list(stk_data.keys())[-1]]
col_sel = ['marketCap', 'trailingPE', 'forwardPE', 'priceToBook']
df_sel = df[col_sel]
