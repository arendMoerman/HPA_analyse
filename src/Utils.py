import cmath
import math
import numpy as np
import struct

# Function to convert strings to lists
def Convert(string, tab=False, use_np=False):
    
    """
    Converts string of floats into list (or array if use_np=True) of floats.
    Per default, splits on space. Can also split on tab delimiters.
    Inputs:
    - string: string of floats to be converted.
    - tab: use tab as line delimiter. Default = False.
    - use_np: convert list of floats to numpy array. Default = False.
    
    Returns either list or numpy array of floats.
    """
    
    string = string.strip()
    if tab == True:
        li = [float(x) for x in string.split("\t")]
    else:
        li = [float(x) for x in string.split(" ")]
    lim = []
    for lli in li:
        lim.append(lli)
    if not use_np:
        return lim
    else:
        return np.array(lim)

def Smooth(a,WSZ):
    
    """
    Python implementation of the MATLAB smooth function. 
    WSZ must be an odd number. If the specified WSZ is even, it will be reduced by one
    to return an odd number.
    Inputs:
    - a: numpy array with data to be smoothed.
    - WSZ: Smoothing window size. Size of moving average filter. Must be odd.
    
    Returns smoothed numpy array.
    """
    
    if (WSZ % 2) == 0:
        WSZ -= 1
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def Binary(num):
    
    """
    Returns string binary representation of floating point numbers.
    Useful for comparing floats.
    """
    
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def Counter(idx, nmax, mod=10000):
    
    """
    Counter, to ensure that a big loop is actually doing stuff.
    """
    
    if idx % mod == 0:
        print("# %.2f / 100..." % ((idx/nmax)*100))
        
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
