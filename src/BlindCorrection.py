import cmath
import math
import matplotlib.pyplot as pt
import src.Utils as ut

import numpy as np
import scipy.signal as sc
import scipy.interpolate as inter
from multiprocessing import Pool
import time

# Need S21, tonebl, toneKID, len(blindlist) and sig. Last two are constants
def ToneCal(S21bl, tonebl, toneKID, bl_length, gamm, lam):
    
    """
    Calculate calibration correction for KIDS.
    """
    
    theta       = -bl_length/2
    length_data = S21bl.shape[0]
    datrange    = np.arange(length_data)
    gX, gY      = np.meshgrid(np.arange(bl_length), datrange) #Create grid for filter
    Gauss2D     = gamm * np.exp((gX+theta)*(gX+theta) * lam) * np.where(gY == round(length_data/2), 1, 0)
    
    filtblind   = sc.convolve2d(S21bl, np.rot90(Gauss2D, k=2), mode="same")
        
    gX, gY      = np.meshgrid(tonebl, datrange)
    gX2, gY2    = np.meshgrid(toneKID, datrange)

    ipreal      = inter.griddata(np.array([gX.ravel(), gY.ravel()]).T, np.real(filtblind).ravel(), np.array([gX2.ravel(), gY2.ravel()]).T, method="linear")
    ipcomp      = inter.griddata(np.array([gX.ravel(), gY.ravel()]).T, np.imag(filtblind).ravel(), np.array([gX2.ravel(), gY2.ravel()]).T, method="linear")
        
    zi          = ipreal.reshape(gX2.shape) + 1j * ipcomp.reshape(gX2.shape)
        
    range_X     = np.array([min(tonebl), max(tonebl)])
    out_range   = np.where(np.logical_or(gX2[0,:] < range_X[0], gX2[0,:] > range_X[1]))

    if len(out_range[0]) != 0:
        adds            = np.zeros((nokids, len(out_range)))
        zi              = np.concatenate(zi, adds, axis=1)
        zi[:,out_range] = np.ones((length_data, len(out_range[0]))) * np.mean(filtblind,axis=1)
        
    return zi

def BlindCorrection(S21, 
                    tone,
                    blindlist,
                    ptwind,
                    colspace,
                    plotty):
    
    """
    This script calibrates the blind tones. Averages over time.
    """
    
    (nfreq, ncols)  = S21.shape
    
    nokids          = ncols / colspace
    blindcor        = np.zeros((nfreq, int(nokids))) 
    blindinterp     = blindcor

    blindlist_usen  = blindlist[np.where((tone[blindlist] < 0) & (tone[blindlist] != -1))]
    KID_list_usen   = np.transpose(np.argwhere(tone < 0))[0]
    blindlist_usep  = blindlist[np.where((tone[blindlist] > 0) & (tone[blindlist] != -1))]
    KID_list_usep   = np.transpose(np.argwhere(tone > 0))[0] 
    
    S21neg          = S21[:,colspace*(blindlist_usen)]
    S21pos          = S21[:,colspace*(blindlist_usep)]
    
    sig             = ptwind
    
    gamm            =  1 / (2 * math.pi * sig*sig)
    lam             = -1 / (2 * sig*sig)
    
    bl_lengthn      = len(blindlist_usen)
    bl_lengthp      = len(blindlist_usep)

    if len(blindlist_usen) != 0 and len(blindlist_usep) != 0:
        zi_neg = ToneCal(S21neg, 
                              tone[blindlist_usen], 
                              tone[KID_list_usen], 
                              bl_lengthn, 
                              gamm,
                              lam)
        zi_pos = ToneCal(S21pos, 
                              tone[blindlist_usep], 
                              tone[KID_list_usep], 
                              bl_lengthp, 
                              gamm,
                              lam)
    
    blindcor = np.zeros(S21.shape, dtype=complex)
    blindcor[:,KID_list_usen] = S21[:,colspace*KID_list_usen] / zi_neg
    blindcor[:,KID_list_usep] = S21[:,colspace*KID_list_usep] / zi_pos
    
    outdata = np.zeros(blindcor.shape, dtype=complex)
    outdata[:,colspace*np.arange(len(tone))] = blindcor
    
    if colspace != 1:
        for i in range(colspace-1):
            outdata[:,colspace*np.arange(len(tone))] = S21arr[:,colspace*np.arange(len(tone))]
        
    refKID = math.ceil(nokids/2)
    if refKID >nokids:
        refKID = nokids
    '''
    fig, ax = pt.subplots(2,1)
    ax[0].plot(np.angle(outdata[:,KID_list_use]))
    
    ax[0].plot(np.angle(blindcor[:,KID_list_use]))
    
    #ax[1].plot(S21
    pt.show()
    '''
    return outdata, KID_list_usen, KID_list_usep, zi_neg, zi_pos
    
    
    
    
