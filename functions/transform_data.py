#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:29:59 2021

@author: pazma
"""

# get dummies
def get_dummies(df,columns):
    import pandas as pd
    return pd.get_dummies(data=df, columns=columns, drop_first=True)