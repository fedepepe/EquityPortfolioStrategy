#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:59:05 2022

@author: federico
"""

import pandas as pd
import tabula
 

stoxxe600_file = 'https://www.stoxx.com/document/Reports/SelectionList/2022/January/sl_sx5e_202201.pdf'
stoxxe600_comp = tabula.read_pdf(stoxxe600_file, pages='all', multiple_tables=False)[0]
symbols = stoxxe600_comp.RIC.tolist()
