#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:59:05 2022

@author: federico
"""

URL = 'https://www.stoxx.com/document/Reports/SelectionList/2023/September/sl_sxxp_202309.pdf'
COMPONENT_FILE_NAME = "./STOXXE600/component_list.txt"


if __name__ == '__main__':
	with open(COMPONENT_FILE_NAME) as file:
		lines = [line.rstrip() for line in file]

	for line in lines:
		matches = line.split()
