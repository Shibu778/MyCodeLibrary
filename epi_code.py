# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:58:55 2021

@author: Shibu Meher
"""

# Code Related to Epilepsy Analysis
import numpy as np

# Function to similarity between two numbers

def numSim(x,y):
    """
    Function to find similarity between two numbers. Takes two numbers
    return a single number. If the we give two series of number of same length,
    then it will give a series of same length with similarity of corresponding
    input series numbers
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    x : float or 1D array of float
        first nubmer
    y : float or 1D array of float
        second number

    Returns
    -------
    float(or 1D array of float), the degree of similarity between the numbers.

    """
    if len(x)!=len(y):
        raise ValueError("Length of the two time series object should be same.")
    
    return 1-np.abs(x-y)/(np.abs(x)+np.abs(y))


# Function to find the mean similarity between time series object

def tSim(X,Y):
    """
    Finds the mean similarity between two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of floats
        First time series object
    Y : 1D array of floats
        Second time series object

    Returns
    -------
    float, mean similarity between the two time series object

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
    
    n = len(X)
    return np.sum(numSim(X,Y))/n


# Function to find the root mean square similarity

def rtSim(X,Y):
    """
    Finds the root mean square similarity between two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object

    Returns
    -------
    float, root mean squared similarity

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
    
    n = len(X)
    return np.sqrt(np.sum(numSim(X,Y)**2)/n)

# Function to find the peak similarity

def pSim(X,Y):
    """
    Finds the peak similarity between two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object

    Returns
    -------
    float, peak similarity

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
        
    n = len(X)
    return np.sum(1-np.abs(X-Y)/(2*np.maximum(np.abs(X),np.abs(Y))))/n

# Function to find the cross correlation or Pearson's Correlation Coefficient

def cross_correlation(X,Y):
    """
    Finds the cross_correlation or Pearson's Correlation Coefficient between
    two time series object of same length
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object

    Returns
    -------
    float, cross correlation or Pearson's Correlation Coefficient

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
    
    m = np.mean(X)
    n = np.mean(Y)
    
    return np.sum((X-m)*(Y-n))/(np.sqrt(np.sum((X-m)**2))*np.sqrt(np.sum((Y-n)**2)))

# Function to find the cosine of angle

def cosine_angle(X,Y):
    """
    Finds the cosine of the angle between the two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object

    Returns
    -------
    float, cosine of angle between the two time series object

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
        
    return np.sum(X*Y)/(np.sqrt(np.sum(X**2))*np.sqrt(np.sum(Y**2)))






















