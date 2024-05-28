import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

thres = 0

def fitGaussAbs(field, x, y, freq):
    global thres
    
    x0 = 30 # as
    y0 = 30
    theta = 0
    
    thres = -11
    
    field_norm = field / np.max(field)
    
    mask_f = 20*np.log10(field_norm) > thres
        
    pars = np.array([x0, y0, theta])
    args = (field_norm, x, y, mask_f)
        
    out = opt.minimize(couplingAbs, x0=pars, args=args, method='L-BFGS-B')
    x0 = out['x'][0]
    y0 = out['x'][1]
    theta = out['x'][2]
    
    ftol = 2.220446049250313e-09 # From scipy docs
    
    tmp_i = np.zeros(len(out.x))
    
    deltas = []
    for i in range(len(out.x)):
        tmp_i[i] = 1.0
        hess_inv_i = out.hess_inv(tmp_i)[i]
        uncertainty_i = np.sqrt(max(1, abs(out.fun)) * ftol * hess_inv_i)
        tmp_i[i] = 0.0
        
        deltas.append(uncertainty_i)
        
    if np.absolute(x0) <= np.absolute(y0):
        ecc = np.sqrt(1 - x0**2/y0**2)
        comfac = (1 - x0**2/y0**2)**(-1/2)
        
        u_e = np.sqrt( (x0 / y0**2 * comfac * deltas[0])**2 + (x0**2 / y0**3 * comfac * deltas[1])**2 )
    else:
        ecc = (x0 - y0) / x0#np.sqrt(1 - y0**2/x0**2)
        comfac = (1 - y0**2/x0**2)**(-1/2)
        
        u_e = np.sqrt( (y0 / x0**2 * comfac * deltas[1])**2 + (y0**2 / x0**3 * comfac * deltas[0])**2 )
    
    #print(out)
    
    idx = np.unravel_index(np.argmax(field, axis=None), field.shape)
    xs = x[idx]
    ys = y[idx]
    
    Psi = GaussAbs(x, y, x0, y0, xs, ys, theta)
    
    eff_mb = np.sum(np.absolute(Psi)**2) / np.sum(np.absolute(field_norm)**2)
    eff_g = np.absolute(np.sum(Psi * field_norm))**2 / (np.sum(np.absolute(field_norm)**2) * np.sum(np.absolute(Psi)**2))
    
    #print('eta_mb = {}, eta_g = {}, ecc = {} +- {}'.format(eff_mb, eff_g, ecc, u_e))
    return out, eff_mb, eff_g, ecc, u_e
        
def couplingAbs(pars, *args):
    x0, y0, theta = pars
    field, x, y, mask_f = args
    field = np.absolute(field)

    idx = np.unravel_index(np.argmax(field, axis=None), field.shape)
    xs = x[idx]
    ys = y[idx]
        
    Psi = GaussAbs(x, y, x0, y0, xs, ys, theta)
    
    #mask_P = 20*np.log10(Psi) > thres

    coupling = np.absolute(np.sum(Psi[mask_f] * field[mask_f]))**2 / (np.sum(np.absolute(field[mask_f])**2) * np.sum(np.absolute(Psi[mask_f])**2))
    #print(coupling)
    
    eta = coupling**2
    eps = 1 - eta
    return eps
        
def GaussAbs(x, y, x0, y0, xs, ys, theta):
    Psi = np.exp(-(((x-xs)/x0*np.cos(theta) + (y-ys)/y0*np.sin(theta)))**2 -(((x-xs)/x0*np.sin(theta) + (y-ys)/y0*np.cos(theta)))**2)
    return Psi 
