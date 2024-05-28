import cmath
import matplotlib.pyplot as pt
import numpy as np
import src.Utils as ut

def FormatData(inputdata,
               CARDparam,
               CALparam,
               Finlist,
               plotty,
               GG):
    """
    Function that formats localsweep data into S21 format.
    Returns S21 data.
    """
    
    LO                      = CARDparam["LO"]
    dumptime                = CARDparam["dumptime"]
    nbins                   = CARDparam["nbins"]
    datatype                = CARDparam["datatype"]
    BW                      = CARDparam["CARDBW"]
    CARDparam["binwidth"]   = BW / nbins
    
    (nfreq, ntone)          = inputdata.shape
    
    if datatype == 1: # Bonn scan of DAC
        ntone                 /= 3
        S21                    = np.zeros((nfreq, 2*ntone), dtype=complex)
        CARDparam["scanwidth"] = (max(trans[0]) - min(trans[0])) * BW
        
        S21     = [[[] for _ in range(nfreq)] for _ in range(int((ntone) * 2))]
        S21t    = ut.Transpose(S21) # Transpose: Cols are freq, rows are tones
        
        for nt in range(int(ntone)):
            CALparam["tone"].append(inputdata[0][(nt - 1) * 3])
            
            for nf in range(nfreq):
                S21t[nf][(nt - 1) * 2]      = inputdata[nf][(nt - 1) * 3] * CARDparam["binwidth"] * LO
                S21t[nf][(nt - 1) * 2 + 1]  = complex(inputdata[nf][(nt - 1) * 3 + 1], inputdata[nf][(nt - 1) * 3 + 2])
    
    elif datatype == 0: # LO scan, 1st col is LO labVIEW, then bin, I, Q, etc. RELEVANT!!
        ntone                   = int((ntone - 1) / 3)
        S21                     = np.zeros((nfreq, 2*ntone), dtype=complex)
        LO                      = inputdata[:,0] * GG
        CARDparam["scanwidth"]  = (max(LO) - min(LO))
        for nt in range(ntone):
            CALparam["tone"].append(inputdata[0,(nt) * 3 + 1])
            
            for nf in range(nfreq):
                S21[nf,nt * 2]  = (inputdata[nf,nt * 3 + 1] + 0.5) * CARDparam["binwidth"] + LO[nf]
                S21[nf,nt * 2 + 1]  = complex(inputdata[nf,nt * 3 + 2], inputdata[nf,nt * 3 + 3])

        CALparam["tone"]        = np.array(CALparam["tone"])      
        Quadrant                = CALparam["tone"]  / np.absolute(CALparam["tone"])
        CALparam["Quadrant"]    = Quadrant
    
    flist = []
    if datatype != 4:
        flist       = np.mean(S21[:,(np.arange(ntone))*2], axis=0)
    else:
        flist       = np.mean(S21[:,(np.arange(ntones))], axis=0)

    return S21, CARDparam, flist, CALparam
    
    
