#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:30:30 2021

@author: pazma
"""

import pandas as pd
import numpy as np

## the correction factor: 
def reweight(pi,q1,r1):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w

def reweight_multi(pi,q,r=1/7):
    w = []
    q_r = [x / r for x in q]
    for n in range(0, len(pi+1)):
        tot = pi.loc[n]*pd.Series(q_r)
        tot_s = sum(tot)
        b = [x / tot_s for x in tot]
        w.append(b)
    w = np.array(w)
    return w