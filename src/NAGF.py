import numpy as np
from scipy import optimize
import src.Structs as st
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.FindCenter as fc

meas_dat = st.meas_dat
GBA_dat = st.GBA_dat

def nested_astiggausian_fit(fit_param, beam, dirmeas, harmIdx, KIDIdx, plot_fit_flag):
    """
    Function that calculates Gaussian beam parameters and efficiences.
    Arguments:
    - fit_param: initial estimates of beam params
    - beam: measured beam and grid
    Returns:
    - fit_vars: fitted parameters
    - efficiencies: beam efficiencies
    """
    
    meas_dat["Xmp"] = beam["x"]
    meas_dat["Ymp"] = beam["y"]
    meas_dat["Zmp"] = np.zeros(beam["x"].shape)
    meas_dat["cpx"] = beam["cdata"]
    meas_dat["lambda"] = beam["lambda"]

    freq = 299792458e3 / beam["lambda"]

    #Idx1, Idx2 = np.unravel_index(np.argmax(np.absolute(beam["cdata"])), beam["cdata"].shape)
    
    #x_center = beam["x"][Idx1, Idx2]
    #y_center = beam["y"][Idx1, Idx2]
    
    x_center, y_center = fc.calcCenter(meas_dat["Xmp"], meas_dat["Ymp"], np.absolute(meas_dat["cpx"]))

    #x_Idx    = np.argmin(np.absolute(beam["x"][0,:] - x_center))
    #y_Idx    = np.argmin(np.absolute(beam["y"][:,0] - y_center))

    id_max = np.unravel_index(np.argmax(np.absolute(beam["cdata"])), beam["cdata"].shape)
    
    M1 = np.absolute(beam["cdata"][id_max])
    ang = np.angle(beam["cdata"][id_max])
    
    args = [M1, 
               ang, 
               fit_param["w0_x"],
               fit_param["w0_y"],
               fit_param["dzxy"],
               fit_param["xi_gb"],
               fit_param["yi_gb"],
               fit_param["zi_gb"],
               fit_param["theta_x0"],
               fit_param["theta_y0"],
               fit_param["theta_z0"]]
    
    optargs = optimize.fmin(func=GBA_eps, x0=args, xtol=1e-10, ftol=1e-10, maxiter=10000, maxfun=10000, disp=False)
    eff = {}
    eff["coupling"] = 1 - GBA_dat["epsilon"]
    
    dx = (np.max(meas_dat["Xmp"]) - np.min(meas_dat["Xmp"])) / (meas_dat["Xmp"].shape[1] - 1)
    dy = (np.max(meas_dat["Ymp"]) - np.min(meas_dat["Ymp"])) / (meas_dat["Ymp"].shape[0] - 1)
    
    a00 = np.sqrt(dx*dy) * np.sum(meas_dat["cpx"] * np.conj(GBA_dat["psi"])) / np.sqrt(np.sum(np.absolute(GBA_dat["psi"])**2))
    
    optargs[0] = np.absolute(a00)
    optargs[1] = np.angle(a00)
    GBA_dat["psi"] = a00 * GBA_dat["psi"]
    
    # Write away psi for warm mirror analysis
    
    if plot_fit_flag:
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
        cmap1 = cm.binary
        cmap2 = cm.twilight
        
        
        im1 = ax[0].pcolormesh(beam["x"], beam["y"], 20*np.log10(np.absolute(beam["cdata"].T) / np.max(np.absolute(beam["cdata"]))), vmin=-30, vmax=0)
        ax[0].contour(beam["x"], beam["y"], 20*np.log10(np.absolute(GBA_dat["psi"].T) / np.max(np.absolute(GBA_dat["psi"]))), levels=[-30, -11, -3], cmap=cmap1, linewidths=0.75, vmin=-30, vmax=0)
        im2 = ax[1].pcolormesh(beam["x"], beam["y"], np.angle(beam["cdata"].T))
        ax[1].contour(beam["x"], beam["y"], np.angle(GBA_dat["psi"].T), levels=1, cmap=cmap2, linewidths=0.75)
        
        ax[0].set_xlabel(r"$x$ / [mm]")
        ax[0].set_ylabel(r"$y$ / [mm]")
        ax[1].set_xlabel(r"$x$ / [mm]")
        ax[1].set_ylabel(r"$y$ / [mm]")
       
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)

        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])
        
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
        c1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
        
        pt.suptitle(r"KID {} Gaussian fit, $f$ = {:.2f} GHz".format(KIDIdx, 2.998e11 / beam["lambda"] / 1e9))
        pt.savefig(fname="images" + dirmeas + "/beams/gauss/lambda_{}".format(harmIdx),bbox_inches='tight', dpi=300)
        pt.close()
        
        fig1, ax1 = pt.subplots(2,1, figsize=(7,5), sharex=True, gridspec_kw={'hspace':0})
        ax1[0].plot(beam["x"][0,:], 20*np.log10(np.absolute(beam["cdata"][int(len(beam["x"][0,:])/2),:]) / np.max(np.absolute(beam["cdata"]))),color='blue',label='data')
        ax1[0].plot(beam["x"][0,:], 20*np.log10(np.absolute(GBA_dat["psi"][int(len(beam["x"][0,:])/2),:]) / np.max(np.absolute(beam["cdata"]))),color='red',label='fit')
        
        ax1[1].plot(beam["x"][0,:], np.angle(beam["cdata"][int(len(beam["x"][0,:])/2),:]),color='blue')
        ax1[1].plot(beam["x"][0,:], np.angle(GBA_dat["psi"][int(len(beam["x"][0,:])/2),:]),color='red')
        
        ax1[0].set_ylabel(r"PNA / [dB]")
        ax1[0].margins(x=0.01)
        ax1[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1[0].legend(frameon=False, prop={'size': 20}, loc='lower center', handlelength=1)
        
        ax1[1].set_xlabel(r"$x$ / [mm]")
        ax1[1].set_ylabel(r"Phase / [rad]")
        ax1[1].margins(x=0.01)
        ax1[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax1[0].set_title(r"KID {} Gaussian fit x-slice, $f$ = {:.2f} GHz".format(KIDIdx, 2.998e11 / beam["lambda"] / 1e9))
        pt.savefig(fname="images" + dirmeas + "/beams/gauss/slices/lambda_{}".format(harmIdx),bbox_inches='tight', dpi=300)
        pt.close()
    
    eff["w0x"]      = optargs[2]
    eff["w0y"]      = optargs[3]
    eff["dzxy"]     = optargs[4]
    eff["xi_gp"]    = optargs[5]
    eff["yi_gp"]    = optargs[6]
    eff["zi_gp"]    = optargs[7]
    eff["theta_x"]  = optargs[8]
    eff["theta_y"]  = optargs[9]
    eff["theta_z"]  = optargs[10]
    

    return optargs, eff

def GBA_eps(args):
    """
    Function to minimize beam parameters, given rotation and translation between beam and measurement 
    coordinate systems.
    """
    
    mag_gb, phs_gb, w0x, w0y, dzxy, xi_gp, yi_gp, zi_gp, theta_x, theta_y, theta_z = args
    
    global meas_dat
    global GBA_dat
    lam = meas_dat["lambda"]
    # Calculate rotation matrices
    rotX = np.array([[1, 0, 0],
                     [0, np.cos(theta_x), -np.sin(theta_x)],
                     [0, np.sin(theta_x), np.cos(theta_x)]])
    
    rotY = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                     [0, 1, 0],
                     [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    rotZ = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                     [np.sin(theta_z), np.cos(theta_z), 0],
                     [0, 0, 1]])
    
    rotAll = np.matmul(rotZ, np.matmul(rotY, rotX))
    
    # Calculate translations
    tX = meas_dat["Xmp"] - xi_gp
    tY = meas_dat["Ymp"] - yi_gp
    tZ = meas_dat["Zmp"] - zi_gp
    
    # Apply full coordinate transform
    Xgb         = rotAll[0,0]*tX + rotAll[0,1]*tY + rotAll[0,2]*tZ
    Ygb         = rotAll[1,0]*tX + rotAll[1,1]*tY + rotAll[1,2]*tZ
    Zgb_x       = rotAll[2,0]*tX + rotAll[2,1]*tY + rotAll[2,2]*tZ
    
    GBA_dat["X"] = Xgb
    GBA_dat["Y"] = Ygb 
    GBA_dat["Z"] = Zgb_x
    
    # z_x and z_y
    Zgb_y = Zgb_x + dzxy
    GBA_dat["zy"] = Zgb_y
    
    # Confocal distance
    z_cx = np.pi * w0x**2 / lam
    z_cy = np.pi * w0y**2 / lam

    GBA_dat["zcx"] = z_cx
    GBA_dat["zcy"] = z_cy
    
    # Beam radius
    wx_gb = w0x * np.sqrt(1 + (Zgb_x / z_cx)**2)
    wy_gb = w0y * np.sqrt(1 + (Zgb_y / z_cy)**2)
    
    GBA_dat["wx"] = wx_gb
    GBA_dat["wy"] = wy_gb
    
    # Radius of curvature
    Rx_gb = Zgb_x + z_cx**2 / (Zgb_x + np.finfo(float).eps)
    Ry_gb = Zgb_y + z_cy**2 / (Zgb_y + np.finfo(float).eps)
    
    GBA_dat["R_x"] = Rx_gb
    GBA_dat["R_y"] = Ry_gb
    
    # Phase along z-axis
    phiX_gb = np.arctan(Zgb_x / z_cx)
    phiY_gb = np.arctan(Zgb_y / z_cy)
    
    GBA_dat["phiX"] = phiX_gb
    GBA_dat["phiY"] = phiY_gb
    
    # Calculate psi
    Psi = np.sqrt(2 / (np.pi * wx_gb * wy_gb)) * np.exp(-(Xgb / wx_gb)**2 - (Ygb / wy_gb)**2 - 1j*2*np.pi / lam * Zgb_x -1j*np.pi/lam * (Xgb**2/Rx_gb + Ygb**2/Ry_gb) + 1j / 2 * (phiX_gb + phiY_gb))
    
    GBA_dat["psi"] = Psi
    
    E = meas_dat["cpx"]
    num = np.sum(E * np.conj(Psi))
    normE = np.sqrt(np.sum(np.absolute(E)**2))
    normP = np.sqrt(np.sum(np.absolute(Psi)**2))
    
    c00 = num / normE / normP
    GBA_dat["c00"] = c00
    
    eta = np.absolute(c00)**2
    GBA_dat["eta"] = eta
    epsilon = 1 - eta
    GBA_dat["epsilon"] = epsilon
    
    return epsilon
    
    
    
    
    
    
    
    
    
