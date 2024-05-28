import math
import cmath
import numpy as np

def CircFit(x, y, seedR):
    """
    This function fits a circle to S21 data by using 
    a least squares algorithm.
    """
    
    divide = np.transpose(np.concatenate(([x], [y], [np.ones(len(x))])))
    radius = -1*(x**2 + y**2)
    a = np.linalg.lstsq(divide,radius, rcond=-1)
    a = a[0]
    
    xc = -0.5*a[0]
    yc = -0.5*a[1]
    alpha = np.angle(xc + 1j*yc)
    R  = seedR * np.sqrt((a[0]**2 + a[1]**2)/4 - a[2])# * np.cos(phi - np.pi / 2)
    
    return xc, yc, R, alpha

