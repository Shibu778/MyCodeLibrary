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

# Function to find the similarity measures between two time series object

def similarity_measures(X, Y, code):
    """
    Finds the required similarity measures encoded by code between the two 
    time series object of same length.
    
    Available similarity measures : mean similarity, root mean square similarity,
    peak similarity, cross correlation or Pearson's correlation coefficient,
    cosine of angle
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object
    code : int
        Code to represent which similarity measure have to determined. Following
        is the the code and corresponding similarity measure associated with it.
        1 : mean similarity
        2 : root mean square similarity
        3 : peak similarity
        4 : Cross Correlation or Pearson's Correlation coefficient
        5 : Cosine of angle

    Returns
    -------
    float, value of required similarity measure.

    """
    if len(X)!=len(Y):
        raise ValueError("Length of the two time series object should be same.")
        
    p = np.mean(X)
    q = np.mean(Y)
    n = len(X)
    
    if code==1:
       return np.sum(numSim(X,Y))/n
   
    elif code==2:
        return np.sqrt(np.sum(numSim(X,Y)**2)/n)
    
    elif code==3:
        return np.sum(1-np.abs(X-Y)/(2*np.maximum(np.abs(X),np.abs(Y))))/n
    
    elif code==4:
        return np.sum((X-p)*(Y-q))/(np.sqrt(np.sum((X-p)**2))*np.sqrt(np.sum((Y-q)**2)))
    
    elif code==5:
        return np.sum(X*Y)/(np.sqrt(np.sum(X**2))*np.sqrt(np.sum(Y**2)))
    
    else:
        ValueError("Please enter the right code. Refer to the documentation.")
        
# Function to calculate all similarity measures

def similarity_measures_all(X,Y):
    """
    Finds all similarity measures between the two 
    time series object of same length.
    
    Available similarity measures : mean similarity, root mean square similarity,
    peak similarity, cross correlation or Pearson's correlation coefficient,
    cosine of angle
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    X : 1D array of float
        First time series object
    Y : 1D array of float
        Second time series object

    Returns
    -------
    array float, containing all similarity measures in the following order
    [mean similarity, root mean square similarity, peak similarity, cross correlation,
     cosine of angle]

    """
    return np.asarray([similarity_measures(X,Y,1),similarity_measures(X,Y,2),
                       similarity_measures(X,Y,3),similarity_measures(X,Y,4),
                       similarity_measures(X,Y,5)], dtype=float)


# Function to plot all similarity measures of a given signal
import pandas as pd
import matplotlib.pyplot as plt

def plot_sm(signal, win_size):
    """
    Plots five similarity measures in a single plot in the following order.
    [mean similarity, root mean square similarity, peak similarity, cross correlation,
     cosine of angle]

    Parameters
    ----------
    signal : 1D array of floats
        Signal after removal of initial artifact
    win_size : int
        Number of points in the signal to consider as the window size

    Returns
    -------
    Returns a pandas dataframe containing similarity measures. Plot a figure
    containing all the similarity measures with respect to window index.

    """
    t_num_win = int(len(signal)/win_size)
    a1 = signal[:win_size]
    sm = []
    for i in range(t_num_win):
        a2 = signal[i*win_size:(i+1)*win_size]
        sm.append(similarity_measures_all(a1,a2))

    sm = np.asarray(sm, dtype=float)
    
    df = pd.DataFrame(sm, columns=['Mean Similarity', 'Root Mean Square Similarity', 'Peak Similarity', 'Cross Correlation', 'Cosine of angle'])
    
    sm1 = sm.T
    plt.figure(1)
    for i in range(5):
        plt.plot(sm1[i])
    
    figure, axes = plt.subplots(nrows=5, ncols=1, figsize = (15,15))
    for i in range(5):
        axes[i].plot(sm1[i])

    figure.tight_layout()
    return df

# Extract and remove data from edf file and remove the initial artifact
from pyedflib import highlevel
import pyedflib

def read_data_from_edf(file_name, which_channel, initial_art):
    """
    Read a particular channel from an edf file and removes the initial artifact.

    Parameters
    ----------
    file_name : string
        Absolute or relative file name
    which_channel : integer
        Index of the channel in the edf file corresponding to a particular channel
    initial_art : int
        Length of the initial artefact to be removed.

    Returns
    -------
    1D array of float : Initial artefact removed signal.

    """
    signals, signal_headers, header = highlevel.read_edf(file_name)
    f = pyedflib.EdfReader(file_name)
    signal_labels = f.getSignalLabels()
    print("Signals Present in the file are : \n")
    print(signal_labels)
    print("\n You have selected the signal :")
    print(signal_labels[which_channel])
    f.close()
    
    a = signals[which_channel][initial_art:]
    
    return a

# Function to directly test similarity measures from an edf file

def test(file_name, which_channel, initial_art, win_size):
    """
    Function to test plot similarity measures directly from filename and channel number.

    Parameters
    ----------
    file_name : string
        Absolute or relative path of the edf file to be read
    which_channel : int
        index of the channel to be read from the edf file
    initial_art : int
        Number of initial artefact points to be removed
    win_size : int
        Number of points to be taken as window size

    Returns
    -------
    pd.Dataframe : A dataframe containing the information about the similarity measures
    of the particular signal. Plot the similarity measures in a group of subplot in the
    following order.
    [mean similarity, root mean square similarity, peak similarity, cross correlation,
     cosine of angle]

    """
    print("File you are reading is : "+file_name)
    sig = read_data_from_edf(file_name, which_channel, initial_art)
    df = plot_sm(sig, win_size)
    print("The order of the subplots is : [mean similarity, root mean square similarity, peak similarity, cross correlation, cosine of angle]")
    
    return df

# Function to calculate Euclidean Distance

def euclidean_distance(T,S):
    """
    Calculates the Euclidean distance between the two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    T : 1D array of float
        First Time series object
    S : 1D array of float
        Second Time Series Object

    Returns
    -------
    float : Distance between the two time series object

    """
    if len(T)!=len(S):
        raise ValueError("Length of the two time series object should be same.")
    
    return np.sqrt(np.sum((T-S)**2))

# Function to calculate Manhattan Distance/Cityblock Distance

def manhattan_distance(T, S):
    """
    Calculates the manhattan distance between two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    T : 1D array of float
        First time series object
    S : 1D array of float
        Second time series object

    Returns
    -------
    float : Manhattan distance between the time series objects

    """
    if len(T)!=len(S):
        raise ValueError("Length of the two time series object should be same.")
        
    return np.sum(np.abs(T-S))

# Function to calculate Maximum Distance

def maximum_distance(T, S):
    """
    Calculates the maximum distance between two time series object of same length.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    T : 1D array of float
        First time series object
    S : 1D array of float
        Second time series object

    Returns
    -------
    float : Maximum distance between the two time series object

    """
    if len(T)!=len(S):
        raise ValueError("Length of the two time series object should be same.")
        
    return np.max(np.abs(T-S))

# Function to calculate Minkowski Distance

def minkowski_distance(T, S, p):
    """
    Calculates Minkowski distance or Lp-norm of order p. Euclidean, Manhattan and 
    maximum distance are instances of minkowski distance of order 2, 1 and infinity,
    respectively.
    
    Ref : http://dx.doi.org/10.5772/49941

    Parameters
    ----------
    T : 1D array of float
        First time series object
    S : 1D array of float
        Second time series object
    p : int
        order of the minkowski distance

    Returns
    -------
    float : Minkowski distance between the two time series object.

    """
    if len(T)!=len(S):
        raise ValueError("Length of the two time series object should be same.")
        
    return np.power(np.sum(np.abs((T-S)**p)), 1/p)


# Function to plot the distance of each points of phase space plot from origin w.r.t. time

def dist_from_origin(x, y, z):
    """
    Calculates the distance of points in a phase space of m-dimension from the origin.
    Plots the distance with respect to time.


    Parameters
    ----------
    x : 1D array of floats
        x-coardinates of points in the phase space
    y : 1D array of floats
        y-coordinate of points in the phase space
    z : 1D array of floats
        z-coordinate of points in phase space

    Returns
    -------
    1D array of floats : distance of the points from the origin

    """
    if len(x)!=len(y) or len(y)!=len(z) or len(x)!=len(z):
        raise ValueError("Length of the three time series object should be same.")
        
    data = np.asarray([x, y, z], dtype = float)
    dist = np.sqrt(np.sum(data**2, 0))
    
    plt.plot(dist)
    
    return dist

# Function to plot the distance of each points(m-dimensional points) of phase space plot from origin w.r.t. time

def dist_from_origin_m_dim(data):
    """
    Calculate the distance of the phase space points from the origin in m dimensional 
    space.

    Parameters
    ----------
    data : N x m dimensional array
        Contains the position of points in m dimensional space.

    Returns
    -------
    1D array : Distance of each m-dimensinal points from the origin.

    """
    dist = np.sqrt(np.sum(data**2, 1))
    
    plt.plot(dist)
    
    return dist


# Function to plot mean similarity, mean and standard deviation on the same plot

def ms_mean_std_plot(signal, win_size):
    """
    Function to calculate the mean similarity with respect to first window, and mean
    and standard deviation of all the windows and plot it in the same plot.

    Parameters
    ----------
    signal : 1D array of floats
        Data to be analyzed
    win_size : int
        window size

    Returns
    -------
    Values of mean similarity, mean and standard deviation and a plot

    """
    
    t_num_win = int(len(signal)/win_size)
    a1 = signal[:win_size]
    sm = []
    for i in range(t_num_win):
        a2 = signal[i*win_size:(i+1)*win_size]
        sm.append([tSim(a1, a2), np.mean(a2), np.std(a2)])
    
    sm = np.asarray(sm, dtype = float)
    sm1 = sm.T
    
    df = pd.DataFrame(sm, columns=['Mean Similarity', 'Mean', 'Standard Deviation'])
    
    plt.figure(1)
    for i in range(3):
        plt.plot(sm1[i])
    
    figure, axes = plt.subplots(nrows=3, ncols=1, figsize = (15,15))
    for i in range(3):
        axes[i].plot(sm1[i])

    figure.tight_layout()
    return df


















