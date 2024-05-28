import math
import cmath
import numpy as np
import scipy.optimize as opt
import scipy.special as sc

import src.Structs as st
import matplotlib.pyplot as pt
import matplotlib.ticker as ticker
import src.Utils as ut

from src.FormatData import FormatData
from src.BlindCorrection import BlindCorrection
from src.PointToFind import PointToFind
from src.CircFit import CircFit
from src.NumDiff import NumDiff

def ThetaAtan(x, theta0, Q, Fr):
    return -theta0 + 2*np.arctan(2*Q*((x - Fr)/Fr))

def FitAtan(f_arr, theta_i, theta0, Q, Fr):
    f_arr = f_arr / 1e9
    p0 = np.array([theta0, Q, Fr/1e9])
    fitresult, fitcov = opt.curve_fit(ThetaAtan, f_arr, theta_i, p0=p0)
    fitresult[2] *= 1e9
    fitcov[2] *= 1e9
    
    return fitresult, fitcov

def SkewLorentz(x, Smin, Q, Fr):
    return 10*np.log10(1-(1-Smin**2)/(1+(2*Q*(x - Fr)/Fr)**2))

def FitS21(data, fres, Qf, S21min):
    sc.seterr(all="ignore")
    data = np.transpose(data)
    fres /= 1e9
    x = data[:,0] / 1e9
    y = data[:,1]**2

    p0 = np.array([S21min, Qf, fres])

    fitresult, fitcov = opt.curve_fit(SkewLorentz, x, 10*np.log10(y), p0 = p0, bounds=([S21min/6, Qf/10, fres*0.9], [S21min*6, Qf*10, fres*1.1]))
    fitresult[2] *= 1e9
    fitcov[2] *= 1e9
    
    return fitresult, fitcov

def MUXfsweep(CARDparam,
              CALparam,
              datadir,
              dirmeas,
              sweepfile,
              diptofit,
              moderange,
              modesweep,
              GG,
              plot_circlefit,
              takephaseref = 1,
              limitrange = 0,
              colspace = 1):
    '''
    Inputs:
    takephaseref = 1: use raw timestream for phase reference
    diptofit: range (in power) of KID to fit. 0.5=3dB points
    refKID: reference KID for deglitching analysis
    '''
    
    refKID      = CARDparam["refKID"]
    poly_fit    = 0
    inftys      = np.where(~np.isfinite(sweepfile[:,3*(refKID - 1)+2]))
    if len(inftys) != 0:
        for i in range(len(inftys)-1):
            dinfs = int(inftys[i+1] - inftys[i])
            if dinfs != 1:
                sweepfile[inftys[i]] = sweepfile[inftys[i+1]]
            else:
                if i != 1:
                    sweepfile[inftys[i]] = sweepfile[inftys[i-1]]
                else:
                    sweepfile[inftys[i]] = sweepfile[inftys[i+2]]
        sweepfile[inftys[-1]] = sweepfile[inftys[-1] - 1]
        
    if limitrange != 0:
        sweepfile = sweepfile[limitrange] # PAY ATTENTION TO INDEXING!!!! if for example 10 in matlab, 9 in python
    
    goodlist                = CARDparam["goodlist"]
    blindlist               = CARDparam["blindlist"]
    smooth_window_size      = CARDparam["smooth_wdow_size"]
    CALparam["sweepfile"]   = sweepfile

    ################ CALL TO FormatData.py ################
    S21, CARDparam, flist, CALparam = FormatData(sweepfile,
               CARDparam,
               CALparam,
               [],
               2,
               GG)
    #######################################################
    CARDparam["tonenumber"] = int(S21.shape[1] / 2)
    nofpts                  = S21.shape[0]
    data                    = st.data
    rtones                  = np.arange(CARDparam["tonenumber"])
    data["comdatasweep"]    = np.zeros((S21.shape[0], int(S21.shape[1]/2)), dtype=complex)

    data["comdatasweep"][:,rtones] = S21[:,2*rtones+1]
    datsweep                = data["comdatasweep"] # So that we dont have to type it everytime

    indexfres               = np.argmin(np.absolute(datsweep[(round(nofpts/2) - diptofit):(round(nofpts/2) + diptofit+1), rtones]),axis=0)
    indexfres               = round(nofpts/2) - diptofit + indexfres 
    
    CALparam["phaseref"]    = np.zeros(len(rtones))
    for i in rtones:
        #CALparam["tone"] = CALparam["Quadrant"] * CALparam["tone"]
        if takephaseref == 1:
            CALparam["phaseref"][i] = np.angle(datsweep[round(nofpts/2),i])
        elif takephaseref == 2:
            CALparam["phaseref"][i] = np.angle(datsweep[indexfres,i])
            
    nofpts              = len(datsweep[:,CARDparam["tonenumber"]-1])
    data["f"]           = np.zeros(datsweep.shape)
    data["f"][:,rtones] = np.absolute(S21[:,2*(rtones)])
    
    blindcorr_input     = datsweep * np.exp(-1j * CALparam["phaseref"])
    
    ################ CALL TO BlindCorrection.py ################
    data["normdatasweep"], _, _, _, _ = BlindCorrection(blindcorr_input,
                    CALparam["tone"],
                    CARDparam["blindlist"],
                    smooth_window_size,
                    1,
                    0)
    #############################################################
    normsweep = data["normdatasweep"] # Shortcut
    seedR = 1
    S21max = np.amax(np.absolute(normsweep), axis=0)

    # Normalization to S21 max
    for i in range(CARDparam["tonenumber"]):
        CALparam["S21max"].append(S21max[i])
        normsweep[:,i] = normsweep[:,i] / S21max[i]
    
    S21min = np.amin(np.absolute(normsweep[(round(nofpts/2)-diptofit-1):(round(nofpts/2)+diptofit-1), np.array(range(CARDparam["tonenumber"]))]), axis=0)
    
    centrefind          = round(len(normsweep)/2)
    
    fitpt               = np.sqrt(S21min**2 + diptofit * (np.ones(len(S21min)) - S21min**2))
    psBpt               = np.zeros(len(S21min))

    dBpt                = np.zeros(len(S21min))
    
    KDP                 = st.KIDparam
    KDP["KIDBW"]        = np.zeros(len(goodlist))
    KDP["Q"]            = np.zeros(len(goodlist))
    KDP["theta0"]       = np.zeros(len(goodlist))
    KDP["phi0"]         = np.zeros(len(goodlist))
    KDP["Qi"]           = np.zeros(len(goodlist))
    KDP["Qc"]           = np.zeros(len(goodlist))
    KDP["Qr"]           = np.zeros(len(goodlist))
    KDP["f_used"]       = np.zeros(len(goodlist))
    
    CALparam["KIDcalphase"] = np.zeros(len(goodlist))
    CALparam["KIDcalreal"] = np.zeros(len(goodlist))
    CALparam["KIDcalR"] = np.zeros(len(goodlist))
    CALparam["xfit"]    = np.zeros(len(goodlist))
    CALparam["yfit"]    = np.zeros(len(goodlist))
    
    badlist = []
    KIDcallibrated = np.zeros((len(normsweep), len(goodlist)), dtype=complex)
    
    total_kids_uncal = np.zeros(0)
    total_kids_cal = np.zeros(0)
    total_freq = np.zeros(0)
    
    for i in range(len(goodlist)):
        KDP["fres"].append(np.absolute(S21[indexfres[(goodlist[i])]-1, (2*(goodlist[i]))]))
        fres = np.array(KDP["fres"])
        KDP["S21min"].append(S21min[goodlist[i]])
    
        if moderange == "points":
            fitrange = np.arange((centrefind-diptofit),(centrefind+diptofit+1))

        f = data["f"]
    
        ptind = PointToFind(f[fitrange,goodlist[i]], np.absolute(normsweep[fitrange,goodlist[i]]), 0, 4)
        
        if isinstance(ptind, int):
            ptind = [ptind]
            
        ptind = ptind + fitrange[0] - 1
        
        minind = np.argmin(np.absolute(ptind - centrefind))

        mindipind = ptind[minind]
        
        S21min[goodlist[i]] = np.absolute(normsweep[mindipind,goodlist[i]])
        S21res = normsweep[mindipind,goodlist[i]]
        dBpt[goodlist[i]] = np.sqrt((S21min[goodlist[i]]**2 + 1)/2)
        
        ptind = []
        boolArr = (fitrange == mindipind)
        min_inrange = [idx for idx, x in enumerate(boolArr) if x == True]
        min_inrange = min_inrange[0]

        if min_inrange > 0:
            ptind.append(np.argmin(np.absolute(np.absolute(normsweep[fitrange[0:min_inrange],goodlist[i]]) - dBpt[goodlist[i]])))
            ptind.append(np.argmin(np.absolute(np.absolute(normsweep[fitrange[(min_inrange+1):-1],goodlist[i]]) - dBpt[goodlist[i]])))
        else:
            ptind.append(0)
            ptind.append(0)

        ptind[0] = fitrange[0] + ptind[0]
        ptind[1] = fitrange[min_inrange+1] + ptind[1] + 1
        
        if ptind[0] <= 1:
            ptind[0] = 1
            
        if ptind[1] >= nofpts:
            ptind[1] = nofpts
        
        ptind = np.array(ptind)
        dumidx = ptind[np.where((ptind-mindipind) >= 0)]
        
        minind = np.argmin(dumidx)
        bwindexup = dumidx[minind]
        
        if not bwindexup:
            bwindex = nofpts
        
        dumidx = ptind[np.where((ptind-mindipind) <= 0)]
        minind = np.argmin(dumidx)
        bwindexdown = dumidx[minind]
        
        if not bwindexdown:
            bwindexdown = 1
        
        dBf = []
        dBf.append(bwindexdown)
        dBf.append(bwindexup)
        
        # Lorentzian fit
        if modesweep == "Lorentz":
            KDP["KIDBW"][goodlist[i]] = f[bwindexup,goodlist[i]] - f[bwindexdown, goodlist[i]]
           
            #if KDP["KIDBW"][goodlist[i]] < 0:
            #    KDP["KIDBW"][goodlist[i]] = f[bwindexdown,goodlist[i]] - f[bwindexup, goodlist[i]]

            if KDP["KIDBW"][goodlist[i]] == 0:
                KDP["KIDBW"][goodlist[i]] = f[-1,goodlist[i]] - f[0,goodlist[i]]
               

            KDP["Q"][goodlist[i]] = KDP["fres"][goodlist[i]] / KDP["KIDBW"][goodlist[i]]
            data_in = np.concatenate(([f[fitrange,goodlist[i]]], [np.absolute(normsweep[fitrange,goodlist[i]])]))
            ##################### CALL TO FITS21 #########################
            fitresult, fitcov = FitS21(data_in, 
                                       KDP["fres"][goodlist[i]], 
                                       KDP["Q"][goodlist[i]], 
                                       KDP["S21min"][goodlist[i]])
            ##############################################################
            
            KDP["Q"][goodlist[i]] = fitresult[1]
            if fitcov[1][-1] > 1e-4:
                badlist.append(i)
                
            KDP["fres"][goodlist[i]] = fitresult[2]
            KDP["S21min"][goodlist[i]] = fitresult[0]
            
            KDP["Qi"][goodlist[i]] = KDP["Q"][goodlist[i]] / KDP["S21min"][goodlist[i]]
            KDP["Qc"][goodlist[i]] = KDP["Qi"][goodlist[i]] * KDP["Q"][goodlist[i]] /( KDP["Qi"][goodlist[i]] - KDP["Q"][goodlist[i]])
            KDP["f_used"][goodlist[i]] = f[centrefind,goodlist[i]]
            
        x = np.real(normsweep[fitrange,goodlist[i]])
        y = np.imag(normsweep[fitrange,goodlist[i]])
        
        ##################### CALL TO CIRCFIT #########################
        xfit, yfit, Rfit, alpha = CircFit(x, y, seedR)
        ###############################################################
        
        reelshift   = np.sqrt(xfit**2 + yfit**2)
        rotation    = alpha
            
        CALparam["KIDcalphase"][goodlist[i]]    = rotation
        CALparam["KIDcalreal"][goodlist[i]]     = reelshift
        CALparam["KIDcalR"][goodlist[i]]        = Rfit
        CALparam["xfit"][goodlist[i]]           = xfit
        CALparam["yfit"][goodlist[i]]           = yfit
        
        if colspace == 2:
            KIDcallibrated[:,2*(goodlist[i]-1)+2] = normsweep[:,2*(goodlist[i]-1)+2]
        
        S21res_cal = ((xfit+1j*yfit) - S21res) * np.exp(-1j*alpha)
        KIDcallibrated[:,goodlist[i]] = ((xfit+1j*yfit) - normsweep[:,goodlist[i]]) * np.exp(-1j*alpha)
        theta0 = 0

        fitresult, _ = FitAtan(f[fitrange,goodlist[i]], 
                               np.angle(KIDcallibrated[fitrange,goodlist[i]]), 
                                theta0, 
                                KDP["Q"][goodlist[i]], 
                                KDP["fres"][goodlist[i]])

        KDP["theta0"][goodlist[i]]  = fitresult[0]
        KDP["Q"][goodlist[i]]       = fitresult[1]
        KDP["fres"][goodlist[i]]    = fitresult[2]
        
        KDP["phi0"][goodlist[i]]    = KDP["theta0"][goodlist[i]] - np.angle(xfit + 1j*yfit)
        KDP["Qc"][goodlist[i]]      = (np.sqrt(xfit**2 + yfit**2) + Rfit) / (2*Rfit) * KDP["Q"][goodlist[i]]

        x1 = np.real(KIDcallibrated[fitrange,goodlist[i]])
        y1 = np.imag(KIDcallibrated[fitrange,goodlist[i]])
        
        fullfreq = (f[:,goodlist[i]] - KDP["fres"][goodlist[i]]) / KDP["fres"][goodlist[i]]
        normfreq = (f[fitrange,goodlist[i]] - KDP["fres"][goodlist[i]]) / KDP["fres"][goodlist[i]]
        thetafit = ThetaAtan(f[fitrange,goodlist[i]], 
                             KDP["theta0"][goodlist[i]], 
                             KDP["Q"][goodlist[i]], 
                             KDP["fres"][goodlist[i]])
        
        uncal = datsweep[fitrange,goodlist[i]] / np.max(np.absolute(datsweep[fitrange,goodlist[i]]))
        idx_min = np.argmin(np.absolute(datsweep[fitrange,goodlist[i]]))
        uncal_min = uncal[idx_min]
        
        total_kids_uncal = np.append(total_kids_uncal, datsweep[:,goodlist[i]] / np.max(np.absolute(datsweep[:,goodlist[i]])))
        total_kids_cal = np.append(total_kids_cal, normsweep[:,goodlist[i]])
        total_freq = np.append(total_freq, f[:,goodlist[i]])

        if plot_circlefit:
            fig, ax = pt.subplots(1,1, figsize=(5,5))
            ax.plot(np.real(uncal), np.imag(uncal), color='black', label='uncalibrated')
            ax.scatter(np.real(uncal_min), np.imag(uncal_min), color='black', zorder=100)
            
            ax.plot(x, y, 'r', label='calibrated')
            
            ax.plot(x1, y1, 'b', label='rotate + shift')

            ax.scatter(np.real(S21res), np.imag(S21res), color='black', zorder=100)
            ax.scatter(np.real(S21res_cal), np.imag(S21res_cal), color='black', zorder=100)
            
            ax.scatter(0, 0, color='black', marker='X', zorder=100)
            ax.scatter(xfit, yfit, color='black', marker='X', zorder=100)
            
            ax.plot([0,np.real(S21res_cal)], [0,np.imag(S21res_cal)], ls='--', c='k', lw=0.7, zorder=0)
            
            ax.tick_params(which="both", direction="in")
            ax.set_box_aspect(1)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.set_ylabel(r"$\Im{(\:S_{21}\:)}$")
            ax.set_xlabel(r"$\Re{(\:S_{21}\:)}$")
            ax.set_aspect(1)
                        
            ax.legend(frameon=False, prop={'size': 12},handlelength=1)
            
            #pt.savefig(fname="images" + dirmeas + "/circle_fits/KID_{}_circle.png".format(i),bbox_inches='tight', dpi=300)
            pt.show()
            pt.close()
    
    return KDP, CALparam, CARDparam, data
