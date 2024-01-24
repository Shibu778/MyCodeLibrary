# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:08:24 2020

@author: Shibu Meher

This code repository stores useful function for chaotic data analysis.
"""


"""
Additional Notes:
    1) Reading x and y by copy paste method. Change the split separator if required.
        x = list(map(float,input("Enter the array x: \n").split("\n")))
        Paste the x data
        y = list(map(float,input("Enter the array y: \n").split("\n")))
        Paste the y data

    2) Reading data from excel file
        import xlrd
        loc = '../Data/Inlet outlet temperature pressure kiln 1.xlsx'
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        sheet.cell_value(0, 0)
        sheet.nrows # number of rows
        sheet.ncols # number of columns
    
"""

####################################################################################

# Function for Linear Interpolation

def linear_interpolation(x, y, m1):
    """
    This function takes two series x and y and Fits y=mx + c line between each point
    and finds out the intermediate value of y for a given intermediate value of x.
    If (x1,y1) and (x2, y2) are two given points. Then the equation of line having
    these two points is (y-y1)=(y2-y1)*(x-x1)/(x2-x1).
    
    Inputs:
        x : an list of length n consisting the x data
        y : an list of length n consisting the y data
        m : number of steps the two points are two be divided
            if the number of steps m = 2 and the two end numbers are 1 and 3,
            then only the new x series is 1,2,3.
        
    Output:
        z1 : an list of length (n-1)*m+1 indecating the time index or x data
        z2 : an list of length (n-1)*m+1 indecating the intepolated y
        
    """
    # Check whether the length of x and y are same or not
    if len(x)!=len(y):
        print("Please Enter x and y of same length.")
        return 0
    n = len(x) #  Calculating the length of x

    # Create variables to store the interpolated x and y
    z1 = []
    z2 = []
    
    # Create a loop for calculating m and c and storing data in z1 and z2
    for i in range(n-1):
        m = (y[i+1]-y[i])/(x[i+1]-x[i]) # Calculating the slope of the line
        c = y[i]-m*x[i] # Calculating the intercept of the line
        dx = (x[i+1]-x[i])/m1 # Calculating the step in x data to be created
        x1 = [x[i]+j*dx for j in range(m1+1)] # Generating the intermediate array of x
        # Calculating the y data for intermediate value of x
        
        y1 = [round(m*x1[j]+c,2) for j in range(m1+1)]
        
        # Storing the x1 and y1 value in z1 and z2
        # Note: the last value of x1 and y1 are not stored as that value will be
        # stored in next iteration
        z1 = z1 + x1[:-1]
        z2 = z2 + y1[:-1]
        
    # Add the last element of x and y in z1 and z2
    z1.append(float(x[-1]))
    z2.append(float(y[-1]))
    return [z1, z2]


#######################################################################################


# Function to find the Pearson Correlation Coefficient between two series
from statistics import mean, stdev

def pearson_correlation_coefficient(x, y):
    """
    This function calculates the Pearson Correlation Coefficient between the two given
    series. When the coefficient is zero, the two series are totally uncorrelated and
    when the coefficient is 1, the two series are correlated positively and when it is
    -1 the series are correlated negatively.
    
    Input:
        x : a numpy array x, 1D sequence
        y : a numpy array y, 1D sequence
    Output:
        R : The Pearson Correlation Coefficient betweent the two sequences
    """
    # Check whether the length of x and y are same or not
    if len(x)!=len(y):
        print("Please Enter x and y of same length.")
        return None
    # Calculating the covariance
    xy = [x[i]*y[i] for i in range(len(x))]
    cov_xy = mean(xy)-mean(x)*mean(y)
    
    # Calculating the standard deviation
    x_std = stdev(x)
    y_std = stdev(y)
    
    # Calculating the pearson correlation coefficient
    r = cov_xy/(x_std*y_std)
    
    return r


######################################################################################

# Function to read the columns of a excel file into the sublist of a list
import xlrd

def read_excel_column(loc, xi, xf, y = [], si = 0):
    """
    This function reads the columns of a excel file and return a list contatining
    sub-list of columns.
    Input:
        loc : Location of the excel workbook
        xi : [ri, ci] row and column index of the initial cell
        xf : [rf, cf] row and column index of the final cell
        y : [y1, y2, ...] index of columns to exclude reading default is []
        si : sheet index, 0
    Output:
        z : a list containing the read columns from the workbook as sublist
    """
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(si)
    z = []
    for j in range(xi[1], xf[1]+1):
        if j not in y:
            z1 = []
            for i in range(xi[0], xf[0]+1):
                z1.append(sheet.cell_value(i,j))
            z.append(z1)
        else:
            print("Column " + str(j) + " has not been read.")
    return z

#####################################################################################

# Function to find the Pearson correlation coefficient between different column
# and store the data in a matrix

def correlation_coefficient_matrix(z):
    """
    This function calculates the pearson correlation coefficent between different
    column of a dataset and store the corresponding value in a list with proper index.
    Input:
        z : a list containing list of equal length inside it, consider each sublist as
            a sequence
    Output:
        r_matrix : matrix containing the correlation coeffiecient between different
                    columns. r_matrix[i][j] indicates the correlation coefficient of
                    column z[i] and z[j]
    """
    n = len(z)
    r_matrix = []
    for i in range(n):
        r1 = [pearson_correlation_coefficient(z[i], z[j]) for j in range(n)]
        r_matrix.append(r1)
    return r_matrix


#####################################################################################


# Function to print a list as a matrix

def print_list_as_table(z):
    """
    This function prints a list as matrix with each sublist as column.
    Input:
        z : input list with equal lenght sublist
    
    """
    n = len(z)
    m = len(z[0])
    s = '...'*3*n+"\n"
    for i in range(m):
        for j in range(n):
            if z[j][i]<0:
                s = s+"|"+" "+str(round(z[j][i], 3))+" "
            else:
                s = s+ "|"+"  "+str(round(z[j][i], 3))+" "
        s = s +"|" + "\n"+ "..."*3*n+"\n"
    print(s)

def print_list_as_table1(z):
    """
    This function prints a list as matrix with each sublist as column. Without design.
    Input:
        z : input list with equal lenght sublist
    
    """
    n = len(z)
    m = len(z[0])
    s = ""
    for i in range(m):
        for j in range(n):
            if z[j][i]<0:
                s = s+" "+str(round(z[j][i], 3))+" "
            else:
                s = s+"  "+str(round(z[j][i], 3))+" "
        s = s +"\n"
    print(s)  


######################################################################################


# Function and class for multiple linear regression
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class OrdinaryLeastSquare(object):
    """
    This objects fits and calculates the coeffiecient for simple or multiple linear
    regression between feature vector X and y. Using the predict function a new entry
    can be predicted. 
    
    If the equation is 
        y = Xb + c where y is a nx1 matrix, b is a mx1 matrix and X is a nxm matrix and c is a scalar
        Then make X and nx(m+1) matrix by adding column of ones at the front. Then use the following to
        calculate b.
        b = (X.T.dot(X)).dot(X.transpose()).dot(y)  ---> resulting shape is (m+1)x1
        The first element of b is the bias and the rest are the coefficent of respective features.
        
    Reference : https://towardsdatascience.com/multiple-linear-regression-from-scratch-in-numpy-36a3e8ac8014
    """
    
    def __init__(self):
        self.coefficients = [] # Variable to store coeffiecients
        self.max_values_x = [] # Stores the max value for each column of X
        self.max_value_y = [] # Stores the max value of target column
        
    # Function to fit the feature X with the target y    
    def fit(self, X, y):
        if len(X.shape)==1: X = self._reshape_x(X)
        X, y = self.normalize_data(X, y)
        X = self._concatenate_ones(X)
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    
    
    # Calculates the prediction using the dot product can deal with multiple entry in rows
    def predict(self, entry):
        """
        This function calculates the out of the model based on the given entries.
        """
        entry = np.array(entry)
        entry = entry/self.max_values_x
        return entry.dot(self.coefficients[1:])+self.coefficients[0]
    
    # Function to calculate the MSE metric for the model (MSE - Mean Squared Error)
    def MSE(self, X_test, y_test):
        """
        Function to calculate the mean square error for test cases.
        Intput:
            X_test : Test feature vector sequence, dimension --> nxm
            y_test : Test target scalar sequence, dimension --> nx1
            
        Output:
            mse : Mean Squared Error value of the prediction by the model
        """
        y_pred = self.predict(X_test)
        y_test = y_test/self.max_value_y
        return np.mean((y_test-y_pred)**2)
    
    def normalize_data(self, X, y):
        """
        Normalize the X and y data by dividing the respective maximum value of each column.
        """
        self.max_values_x = np.max(X, axis = 0)
        self.max_value_y = np.max(y, axis=0)
        return X/self.max_values_x, y/self.max_value_y
    
    # Plotting the predicted versus actual value of target
    def plot_pred_vs_actual(self, X, y):
        """
        Function to scatter plot the predicted value versus the actual value.
        Input:
            X : Feature vector
            y : actual target
            
        """
        y_pred = self.predict(X)
        y = y/self.max_value_y
        plt.plot(y, y_pred, 'o', color='black')
        plt.plot(y, y, '-', color='red')
        plt.xlabel(r"y_actual")
        plt.ylabel(r"y_predicted")
        plt.title(r"y_pred versus y_actual, $R^2$={0}".format(self.squared_pearson_corrcoef(X, y)))
        
    def squared_pearson_corrcoef(self, X, y):
        """
        Calculates the Squared Pearson Correlation Coefficient between the predicted and actual value of target.
        This value of coefficient tells accuracy of fitting. If it is closer to 1, then the fitting is good.
        If it is closer to zero, fitting is poor.
        """
        y_pred = self.predict(X)
        y = y/self.max_value_y
        coeff = (np.mean(y*y_pred)-np.mean(y)*np.mean(y_pred))/(np.std(y)*np.std(y_pred))
        print("The Pearson Correlation Coefficient between the predicted and actual value is : {0}"
              .format(coeff**2))
        return coeff**2
        
    
    def print_shape_of_coefficients(self):
        print(type(self.coefficients))
        print(self.coefficients.shape)
        print("Give entries of shape m x {0}".format(self.coefficients.shape[0]-1))

    def _reshape_x(self, X):
        return X.reshape(-1,1)
    
    def _concatenate_ones(self, X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones, X), axis=1)


####################################################################################################

# Function to determine the time delay
# Autocorrelation method
import numpy as np

def acorr(x, maxtau=None, norm=True, detrend=True):
    """
    Returns the autocorrelation of the given scalar time series.
    Calculates the autocorrelation of the given scalar time series as a function of time lag
    using Wiener-Kinchin Theorem.
    
    Parameters
    -----------
    x : array_like
        1D real time series of length N
    maxtau : int, optional (default is N)
        Return the autocorrelation only upto this time delay
    norm : bool, optional (default is True)
        Normalize the autocorrelation so that it is equal to 1 for zero time delay.
    detrend : bool, optional (default is True)
        Substract the mean from the time series (i.e. Constant detrend).This is done so that for
        uncorrelated data, autocorrelation vanishes for all non zero values of time delay.
        
    Returns
    --------
    r : array
        Array of correlation upto maxtau
    
    References : https://github.com/manu-mannattil/nolitsa
    """
    x = np.asarray(x)
    N = len(x)
    
    if not maxtau:
        maxtau = N
    else:
        maxtau = min(N, maxtau)
    
    if detrend:
        x = x - np.mean(x)
        
    # We have to zero pad the data to give it a length 2N - 1.
    # See: http://dsp.stackexchange.com/q/1919
    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))
    
    if norm:
        return r[:maxtau] / r[0]
    else:
        return r[:maxtau]
    

#####################################

# Function to calculate time delay based on Mutual Information

def mi(x, y, bins=64):
    """Calculate the mutual information between two random variables.

    Calculates mutual information, I = S(x) + S(y) - S(x,y), between two
    random variables x and y, where S(x) is the Shannon entropy.

    Parameters
    ----------
    x : array
        First random variable.
    y : array
        Second random variable.
    bins : int
        Number of bins to use while creating the histogram.

    Returns
    -------
    i : float
        Mutual information.
        
        
    References : https://github.com/manu-mannattil/nolitsa
    """
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.  Also, in the limit
    # p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y

def dmi(x, maxtau=1000, bins=64):
    """Return the time-delayed mutual information of x_i.

    Returns the mutual information between x_i and x_{i + t} (i.e., the
    time-delayed mutual information), up to a t equal to maxtau.  Based
    on the paper by Fraser & Swinney (1986), but uses a much simpler,
    albeit, time-consuming algorithm.

    Parameters
    ----------
    x : array
        1-D real time series of length N.
    maxtau : int, optional (default = min(N, 1000))
        Return the mutual information only up to this time delay.
    bins : int
        Number of bins to use while calculating the histogram.

    Returns
    -------
    ii : array
        Array with the time-delayed mutual information up to maxtau.

    Notes
    -----
    For the purpose of finding the time delay of minimum delayed mutual
    information, the exact number of bins is not very important.
    
    References : https://github.com/manu-mannattil/nolitsa
    """
    N = len(x)
    maxtau = min(N, maxtau)

    ii = np.empty(maxtau)
    ii[0] = mi(x, x, bins)

    for tau in range(1, maxtau):
        ii[tau] = mi(x[:-tau], x[tau:], bins)

    return ii

    
#################################################################################################### 


# Function to reconstruct the time-delayed vectors from a time series

def reconstruct(x, dim=1, tau=1):
    """Construct time-delayed vectors from a time series.

    Constructs time-delayed vectors from a scalar time series.

    Parameters
    ----------
    x : array
        1-D scalar time series.
    dim : int, optional (default = 1)
        Embedding dimension.
    tau : int, optional (default = 1)
        Time delay

    Returns
    -------
    ps : ndarray
        Array with time-delayed vectors.
        
    References : https://github.com/manu-mannattil/nolitsa
    """
    m = len(x) - (dim - 1) * tau
    if m <= 0:
        raise ValueError('Length of the time series is <= (dim - 1) * tau.')

    return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])


##################################################################################################

# Function to calculate the distance between two points in a menifold (multi dimensional space)
from scipy.spatial import distance

def dist(x, y, metric='chebyshev'):
    """Compute the distance between all sequential pairs of points.

    Computes the distance between all sequential pairs of points from
    two arrays using scipy.spatial.distance.

    Paramters
    ---------
    x : ndarray
        Input array.
    y : ndarray
        Input array.
    metric : string, optional (default = 'chebyshev')
        Metric to use while computing distances.

    Returns
    -------
    d : ndarray
        Array containing distances.
        
    References : https://github.com/manu-mannattil/nolitsa
    """
    func = getattr(distance, metric)
    return np.asarray([func(i, j) for i, j in zip(x, y)])



##################################################################################################

# Function to compute the average displacement fromt he diagonal of the phase space 

def adfd(x, dim=1, maxtau=100):
    """Compute average displacement from the diagonal (ADFD).

    Computes the average displacement of the time-delayed vectors from
    the phase space diagonal which helps in picking a suitable time
    delay (Rosenstein et al. 1994).

    Parameters
    ----------
    x : array
        1-D real time series of length N.
    dim : int, optional (default = 1)
        Embedding dimension.
    maxtau : int, optional (default = 100)
        Calculate the ADFD only up to this delay.

    Returns
    -------
    disp : array
        ADFD for all time delays up to maxtau.
        
    References : https://github.com/manu-mannattil/nolitsa
    """
    disp = np.zeros(maxtau)
    N = len(x)

    maxtau = min(maxtau, int(N / dim))

    for tau in range(1, maxtau):
        y1 = reconstruct(x, dim=dim, tau=tau)

        # Reconstruct with zero time delay.
        y2 = x[:N - (dim - 1) * tau]
        y2 = y2.repeat(dim).reshape(len(y2), dim)

        disp[tau] = np.mean(dist(y1, y2, metric='euclidean'))

    return disp

################################################################################################














    
    
    
    
    
    
    
    
    
