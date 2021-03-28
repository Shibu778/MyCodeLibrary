# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:03:12 2021

@author: Shibu Meher
"""

# Programs for nonlinear dynamics course in complexity explorer

# Logistic map program

import numpy as np
def logistic_map(x0, r, n):
    """
    This function iterate through a logistic map with parameter r, n times
    starting from the initial condition x0. It returns an array containing the
    value of x as it iterates through the logistic map.

    Parameters
    ----------
    x0 : float
        Initial state of the logistic map
    r : float (must be between 0 and 4)
        Parameter of the logistic map
    n : int
        Number of iteration to be done

    Returns
    -------
    l : the array containg the states covered during evolution.

    """
    x = r*x0*(1-x0) # First iteration
    l = [x]         # Storing the result of first iteration in l
    for i in range(n-1):
        x = r*x*(1-x)
        l.append(x)
    
    return np.asarray(l, dtype=float)




