import math
import cmath
import numpy as np
from src.NumDiff import NumDiff

def PointToFind(x, y, pointvalue, span):
    dif = NumDiff(x, y, span)
    dif2= NumDiff(dif[:,0], dif[:,1], span)
    
    npts = len(x)
    
    boolArr = (dif[1:npts-1,1] > 0) & (dif[0:npts-2,1] < 0)
    
    peaksind = [idx for idx, x in enumerate(boolArr) if x == True]
    
    peaky = []
    peakgroup = 1
    if (peaksind):
        peaky.append(peaksind[0])
        
        for i in range(1,len(peaksind)):
            if peaksind[i] == peaksind[i-1]+1:
                peaky[peakgroup] = [peaky[peakgroup], peaksind[i]]
            else:
                peakgroup += 1
                peaky.append(peaksind[i])
        
        ptind = []
        for i in range(peakgroup):
            ptind.append(np.argmax(dif2[peaky[i],1]))
            ptind[i] = peaky[i]
        
    else:
        ptind = round(len(x)/2)
    
    return ptind
    
    
        
