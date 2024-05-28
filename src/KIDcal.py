import math
import cmath
import numpy as np

def KIDcal(raw, goodlist, CALparam, colspace):
    """
    Function to apply calibration to KID data.
    Rotates and translates KID circle to origin.
    """
    
    KIDcalibrated = np.zeros((len(raw), len(goodlist)), dtype=complex)

    if colspace == 2:
        KIDcalibrated[:,2*(goodlist-1)+2] = raw[:,2*(goodlist-1)+2]
        
    KIDcalibrated[:,colspace*(goodlist-1)+colspace] = (raw[:,colspace*(goodlist-1)+colspace] * np.exp(-1j*CALparam["KIDcalphase"][goodlist]) - CALparam["KIDcalreal"][goodlist]) / CALparam["KIDcalR"][goodlist]
    
    return KIDcalibrated
