import math
import cmath
import numpy as np
import src.Utils as ut

def NumDiff(x, y, span):
    
    """
    Calculates numerical derivatives using symmetric difference quotient and smoothes calculated data.
    Inputs:
    - x: Numpy array containing the independent variable.
    - y: Numpy array containing the variable to be differentiated.
    - span: Smoothing window size to smooth calculated data over.
    
    Returns: Smoothed (N, 1) numpy array containing independent variable and first order derivative [x, dy/dx].
    """

    dx = x[1] - x[0]
    
    dif = np.zeros((len(y),2))

    dif[:,0] = x
    for i in range(1, len(y)-1):
        dif[i,1] = (y[i+1] - y[i-1])/(2*dx)
    
    dif[0,1] = dif[1,1]
    dif[len(y)-1,1] = dif[len(y)-2,1]
    dif[:,1] = ut.Smooth(dif[:,1], span)

    return dif
