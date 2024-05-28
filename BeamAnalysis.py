import math
import time
import os
import sys
import numpy as np
import scipy.special as sc
import scipy.interpolate as interp
import scipy.optimize as opt
import scipy.fft as ft
import matplotlib.pyplot as pt
import pickle
from multiprocessing import cpu_count
import warnings
pt.rcParams['xtick.top'] = True
pt.rcParams['ytick.right'] = True                                                                                                                                               

pt.rcParams['xtick.direction'] = "in"
pt.rcParams['ytick.direction'] = "in"

pt.rcParams['xtick.minor.visible'] = True
pt.rcParams['ytick.minor.visible'] = True
from tqdm import tqdm

import matplotlib
import cmocean

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import src.Structs as st
import src.Utils as ut
from src.NumDiff import NumDiff

from src.MUXfsweep import MUXfsweep
from src.XYscan import XYscan
from src.NAGF import nested_astiggausian_fit
import src.FindCenter as fc
from src.CustomLogger import CustomLogger

from CreatePaths import CreatePaths
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sc.seterr(all="ignore")

clog_mgr = CustomLogger(os.path.basename(__file__))
clog = clog_mgr.getCustomLogger()

matplotlib.rcParams.update({'font.size': 20})
###########################################################################
#                           GLOBAL CONSTANTS                              #
###########################################################################
h                       = 6.626e-34
k                       = 1.3806e-23
c                       = 2.998e8

threshold               = 1e-3
nrptsfit                = 20
num_cores               = 11#cpu_count() - 1
sep                     = os.sep

###########################################################################
#                            CONTROL PARAMS                               #
###########################################################################
chopFlag                = 0
readData                = 0
XYscanFlag              = chopFlag or readData
Grid_data               = 1
find_center             = 1
fit_gauss               = 1

plot_beams              = 0
plot_beams_max          = 1
plot_fit_flag           = 1
plot_gauss_param        = 1

###########################################################################
#                             FILEPATHS                                   #
###########################################################################

dirmeas = "/XYZScan_20230111_194931" # in front of cryo/
path = "/home/arend/Projects/Software/HPAnalysis/measurements"

dirdat = path + dirmeas
for file in os.listdir(dirdat):
    if file.endswith(".fits"):
        fitsname = file

CreatePaths(dirmeas)

###########################################################################
#                             XYT CONFIG                                  #
###########################################################################
XYT                     = st.xy_settings
CDP                     = st.CARDparam

with open(dirdat + sep + "measurement.conf", "r") as file:
    for line in file:
        line = line.split(" ")
        
        if line[0] == "Xcentre":
            x_center = float(line[2])
        
        if line[0] == "Ycentre":
            y_center = float(line[2])

        if line[0] == "nFramesXY":
            XYT["nrtr"] = int(line[2])
        
        if line[0] == "framelen":
            CDP["framelen"] = int(line[2])

        if line[0] == "averages":
            CDP["smooth_wdw_size"] = int(line[2])

XYT["ref_multiplier"]   = 21
#XYT["ref_range"]        = np.arange(21,32)
XYT["ref_range"]        = np.arange(21,41)
XYT["mod_frac"]         = 5
#f_RF                    = 9.8981818e9 * XYT["ref_range"] # Setting ref frequencies
f_RF                    = 9.76547619e9 * XYT["ref_range"] # Setting ref frequencies

CDP["dumptime"]         = 1 / (2e9/2**CDP["framelen"]/CDP["smooth_wdw_size"]) # This is dt for fft
CDP["nbins"]            = 2**CDP["framelen"]
CDP["CARDBW"]           = 2000e6
###########################################################################
#                                BADLIST                                  #
###########################################################################
#print(CDP["dumptime"])
## LT 263
#badlist = np.array([217, 74, 60, 139, 65, 98, 265, 208, 122, 170, 188, 140, 145, 240, 289, 311, 323, 128]) # Fine, corr
badlist = np.array([0, 1, 22, 68, 277, 230, 282, 161, 20, 23, 50, 44, 110, 120, 114, 217, 74, 60, 139, 65, 98, 265, 208, 122, 170, 188, 140, 145, 240, 289, 311, 323, 326, 128, 327]) # Fine, uncor
#badlist = np.array([161, 339, 284, 47, 288, 117, 303, 168, 88, 268, 327, 129, 68, 230, 282, 20, 50, 44, 120, 231, 266, 123, 114, 217, 74, 60, 139, 65, 98, 265, 208, 122, 170, 188, 140, 145, 240, 289, 311, 323, 326, 128]) # Fine, uncor
wideband = np.arange(333, 341) # Wideband KIDs

## LT 223
#badlist = np.array([6, 166, 182, 195, 211, 246])
#wideband = np.array([302, 303, 304, 305, 306, 307]) # Wideband KIDs previous chip
###########################################################################
#                             MAIN ROUTINE                                #
###########################################################################

def Main():
    global x_center
    global y_center
    clog.info(f"STARTED ANALYSIS OF {dirmeas}, {fitsname}")
    clog.info("{} harmonics to analyze".format(len(XYT["ref_range"]), [round(x/1e9,1) for x in f_RF]))
    start_time          = time.time()

    # Initialize  xy list
    with open(dirdat + sep + "xytrace.txt","r") as xy:
        for idx, line in enumerate(xy):
            if idx == 0:
               continue
            else:
                line = ut.Convert(line)
                XYT["xy_trace"].append(line)

    XYT["xy_trace"]     = np.array(XYT["xy_trace"])
    XYT["nrpt"]         = len(XYT["xy_trace"])
    XYT["f_chopper"]    = 1 / CDP["dumptime"] / XYT["mod_frac"] / XYT["ref_multiplier"]
    XYT["end_sett"]     = 0
    XYT["start_sett"]   = 0
    XYT["subchunks"]    = 1
    XYT["trace_mult"]   = 4
    XYT["nbtr"]         = XYT["ref_multiplier"] * XYT["mod_frac"] * XYT["trace_mult"]
    XYT["nr_in_memory"] = XYT["nbtr"] * 10
    XYT["meas"]         = "f_resp"
    XYT["peakpos_ref"]  = 1 + np.round(XYT["f_chopper"] * (XYT["nbtr"]-XYT["start_sett"]-XYT["end_sett"]) * CDP["dumptime"])/XYT["subchunks"]
    XYT["peakpos"]      = 1 + np.round(XYT["f_chopper"] * (XYT["nbtr"]-XYT["start_sett"]-XYT["end_sett"]) * CDP["dumptime"] * XYT["ref_range"])/XYT["subchunks"]
    XYT["ref_set"]      = "abs"

    clog.info("Number of XYZ points: {}".format(XYT["nrpt"]))
    nr_LO = 1 # Number of synthesizer tones
    #print(XYT["f_chopper"])
    ################################# MAIN ANALYSIS LOOP #######################################
    for i in range(nr_LO):
        usedKIDS    = []
        posKIDs     = 0
        posblinds   = 0
        LO          = 0
        with open(dirdat + sep + "kids.list","r") as kids:
            for idx, line in enumerate(kids):
                if idx == 0:
                    LO          = line[4:-1]
                if line == "#KIDs\n": # Save 1st idx of KIDs array
                    posKIDs     = idx
                elif line == "#blinds\n":
                    posblinds   = idx # Save 1st idx of blinds array
                elif line[0] == "#":
                    continue
                else:
                    line = ut.Convert(line, tab=True)
                    usedKIDS.append(line)
        usedKIDS = np.transpose(np.array(usedKIDS))

        sweepfile = []
        with open(dirdat + sep + "localsweep.sweep","r") as sweep:
            for line in sweep:
                line = ut.Convert(line)
                sweepfile.append(line)

        sweepfile = np.array(sweepfile)

        CDP["tonenumber"] = len(usedKIDS[0,:])
        CDP["nr_blinds"]  = CDP["tonenumber"] - (posblinds - 3)
        CDP["nr_kids"]    = CDP["tonenumber"] - CDP["nr_blinds"]
        CDP["nr_refs"]    = 0
        CDP["LO"]         = LO
        CDP["KIDbins"]    = usedKIDS[0,(posKIDs-2):(posblinds-3)]
        CDP["blindbins"]  = usedKIDS[0,(posblinds-3):]
        CDP["refbins"]    = CDP["KIDbins"][0:CDP["nr_refs"]]
        CDP["KID_p"]      = usedKIDS[1,(posKIDs-2):(posblinds-3)]
        CDP["blind_p"]    = usedKIDS[1,(posblinds-3):]
        CDP["p"]          = (np.average(CDP["KID_p"]) + np.average(CDP["blind_p"])) / 2
        CDP["goodlist"]   = CDP["nr_refs"] + np.arange(0, CDP["nr_kids"])
        CDP["blindlist"]  = CDP["nr_kids"] + CDP["nr_refs"] + np.arange(0, CDP["nr_blinds"])
        CDP["ref_tones"]  = np.arange(0, CDP["nr_refs"])
        CDP["refKID"]     = 137 - 1 # -1 explicitly given to comply with 0 based indexing
        if CDP["refKID"] > CDP["goodlist"][-1]:
            CDP["refKID"] = CDP["goodlist"][-1]

        clog.info("Number of KIDs: {}".format(CDP["nr_kids"]))
        clog.info("Number of Blind tones: {}".format(CDP["nr_blinds"]))
        clog.info("Total amount of tones: {}".format(CDP["tonenumber"]))
        clog.info("Frames in XY point: {}".format(XYT["nrtr"]))

        CLP     = st.CALparam

        CDPcopy = {}
        CDPcopy = CDP

        ######################### CALL TO MUXFsweep ###############################
        KID_sweep, CAL_sweep, CARD_sweep, procdata = MUXfsweep(CDPcopy,
                  CLP,
                  dirdat,
                  dirmeas,
                  sweepfile,
                  nrptsfit,
                  "points",
                  "Lorentz",
                  1e6,
                  )
        ###########################################################################

        CDP["scanwidth"] = CARD_sweep["scanwidth"]

        f                   = procdata["f"]
        fres                = KID_sweep["fres"]
        nbfpts              = f.shape[0]
        goodlist            = CDP["goodlist"]

        ######################### CALL TO XYscan ##################################
        if XYscanFlag:
            read_time = time.time()
            XYscan(procdata, CAL_sweep, CDP, XYT, dirdat, dirdat +"/"+ fitsname, 
                    KID_sweep, num_cores, dirmeas, chopFlag, readData, clog)
            clog.info("Readtime = {} seconds using {} cores.".format( time.time() - read_time, num_cores))
        ###########################################################################

    if Grid_data:
        for j in range(num_cores):
            # Load x, y, z, xy and beam_ref_flag into separate containers
            xfile = np.load("temp/x/x_{}.npy".format(j))
            yfile = np.load("temp/y/y_{}.npy".format(j))
            zfile = np.load("temp/z/z_{}.npy".format(j))
            beam_ref_file = np.load("temp/beam_ref/beam_ref_{}.npy".format(j))
            xyfile = np.load("temp/xy/xy_{}.npy".format(j))

            if j == 0:
                x = xfile
                y = yfile
                z = zfile
                beam_ref_flag = beam_ref_file.astype(int)
                xy = xyfile
            else:
                x = np.concatenate((x, xfile))
                y = np.concatenate((y, yfile))
                z = np.concatenate((z, zfile))
                beam_ref_flag = np.concatenate((beam_ref_flag, beam_ref_file.astype(int)))
                xy = np.concatenate((xy, xyfile), axis=2)

        x_uniq = np.unique(x)
        y_uniq = np.unique(y)
        z_uniq = np.unique(z)

        x_gridded = []
        y_gridded = []
        z_gridded = []
        z_gridded_offset = np.zeros((len(y_uniq), len(x_uniq)))

        use_ref = 0

        # Lists to contain beams. Each list entry is a separate KID
        beam_gridded_l           = []
        beam_gridded_offset_l    = []
        beam_gridded_ref_l       = []

        for i in range(nr_LO):
            peak_f = y_gridded
            clog.info("Gridding beam patterns")
            for j in tqdm(CDP["goodlist"], ncols=100, colour="green", total=len(CDP["goodlist"])):
                beam_gridded = np.zeros((len(XYT["ref_range"]), len(x_uniq), len(y_uniq)), dtype=complex)
                beam_gridded_offset = beam_gridded

                pos = np.argwhere(beam_ref_flag)
                z_ref = z[pos]
                x_ref = x[pos]

                beam_gridded_ref = np.zeros(len(pos))
                beam_gridded_ref = xy[1,j,pos] # Extract xy values for reference scans

                if np.sum(np.where(beam_ref_flag)) == 0 or use_ref == 0:
                    ref_sig = np.ones(len(x_ref))
                else:
                    ref_sig = np.absolute(beam_gridded_ref) / np.mean(np.absolute(beam_gridded_ref)) * np.exp(1j*(np.angle(beam_gridded_ref) - np.mean(np.angle(beam_gridded_ref))))

                for xIdx in range(len(x_uniq)):
                    for zIdx in range(1):
                        pos = np.argwhere(np.logical_and.reduce((x.astype(int) == int(x_uniq[xIdx]), np.round(z,4) == np.round(z_uniq[zIdx],4), beam_ref_flag != 1)))

                        if np.sum(np.where(beam_ref_flag)) == 0:
                            ref_pos = xIdx
                            ref_cor = 1
                        else:
                            ref_pos = (xIdx)/2+(zIdx)*(len(x_uniq)-1)/2

                            inter_amp = interp.interp1d(np.arange(len(ref_sig)), np.absolute(ref_sig))(ref_pos)
                            inter_ang = interp.interp1d(np.arange(len(ref_sig)), np.angle(ref_sig))(ref_pos)

                            ref_cor = inter_amp * np.exp(1j*inter_ang)
                        beam_gridded[:,:,xIdx] = np.squeeze(xy[:,j,pos]) / ref_cor

                        if j == 0:
                            if xIdx == 0:
                                y_gridded = np.expand_dims(np.squeeze(y[pos]), axis=1)
                                x_gridded = np.expand_dims(np.squeeze(x[pos]), axis=1)
                                z_gridded = np.expand_dims(np.squeeze(z[pos]), axis=1)

                            else:
                                y_gridded = np.concatenate((y_gridded, y[pos]), axis=1)
                                x_gridded = np.concatenate((x_gridded, x[pos]), axis=1)
                                z_gridded = np.concatenate((z_gridded, z[pos]), axis=1)
                    
                    if len(z_uniq) != 1:
                        for zIdx in range(1,2):
                            ref_pos = (xIdx)/2+(zIdx)*(len(x_uniq)-1)/2

                            inter_amp = interp.interp1d(np.arange(len(ref_sig)), np.absolute(ref_sig))(ref_pos)
                            inter_ang = interp.interp1d(np.arange(len(ref_sig)), np.angle(ref_sig))(ref_pos)

                            ref_cor = inter_amp * np.exp(1j*inter_ang)

                            pos = np.argwhere(np.logical_and.reduce((x.astype(int) == int(x_uniq[xIdx]), np.round(z,4) == np.round(z_uniq[zIdx],4), beam_ref_flag != 1)))

                            beam_gridded_offset[:,:,xIdx] = np.squeeze(xy[:,j,pos]) / ref_cor

                            if j == 0:
                                z_gridded_offset[:,xIdx] = np.squeeze(z[pos])
                
                beam_gridded_l.append(beam_gridded)
                beam_gridded_offset_l.append(beam_gridded_offset)
                beam_gridded_ref_l.append(beam_gridded_ref)

        np.save("temp/beams/beam_gridded_l.npy",beam_gridded_l, allow_pickle=True)
        np.save("temp/beams/beam_gridded_offset_l.npy",beam_gridded_offset_l, allow_pickle=True)
        np.save("temp/beams/beam_gridded_ref_l.npy",beam_gridded_ref_l, allow_pickle=True)

        np.save("temp/x/x_uniq.npy",x_uniq, allow_pickle=True)
        np.save("temp/y/y_uniq.npy",y_uniq, allow_pickle=True)
        np.save("temp/z/z_uniq.npy",z_uniq, allow_pickle=True)

        np.save("temp/x/x_gridded.npy",x_gridded, allow_pickle=True)
        np.save("temp/y/y_gridded.npy",y_gridded, allow_pickle=True)
        np.save("temp/z/z_gridded.npy",z_gridded, allow_pickle=True)
        np.save("temp/z/z_gridded_offset.npy",z_gridded_offset, allow_pickle=True)

    else:
        try:
            beam_gridded_l = np.load("temp/beams/beam_gridded_l.npy")
            beam_gridded_offset_l = np.load("temp/beams/beam_gridded_offset_l.npy")
            beam_gridded_ref_l = np.load("temp/beams/beam_gridded_ref_l.npy")

            x_uniq = np.load("temp/x/x_uniq.npy")
            y_uniq = np.load("temp/y/y_uniq.npy")
            z_uniq = np.load("temp/z/z_uniq.npy")

            x_gridded = np.load("temp/x/x_gridded.npy")
            y_gridded = np.load("temp/y/y_gridded.npy")
            z_gridded = np.load("temp/z/z_gridded.npy")
            z_gridded_offset = np.load("temp/z/z_gridded_offset.npy")
        except:
            clog.error("No beamdata to read! Read .FITS data first.")
            clog.info("Elapsed time: {} seconds".format(time.time() - start_time))
            return 1

    beam_corrected_l = []
    beam_error_l = []
    max_beam_l = []
    sum_beam_l = []

    for i in range(nr_LO):
        for j in CDP["goodlist"]:
            beam_corrected = (beam_gridded_l[j] + beam_gridded_offset_l[j]*np.exp(1j*np.pi/2))/2
            #beam_corrected = (beam_gridded_offset_l[j]*np.exp(1j*np.pi/2))/2
            beam_error = (beam_gridded_l[j] + beam_gridded_offset_l[j]*np.exp(-1j*np.pi/2))/2
            max_beam = np.max(beam_corrected, axis=(1,2))
            sum_beam = np.mean(beam_corrected[:, 25:35, 25:35], axis=(1,2))
            #sum_beam = beam_corrected[:, 30, 30]
            beam_corrected_l.append(beam_corrected)
            beam_error_l.append(beam_error)
            max_beam_l.append(max_beam)
            sum_beam_l.append(sum_beam)

        max_beam_l = np.array(max_beam_l)
        sum_beam_l = np.array(sum_beam_l)

        # Wideband normalization
        wideband_norm = np.mean(np.absolute(max_beam_l[wideband,:]), axis=0)

        # Safe to delete uncorrected lists I guess #
        del beam_gridded_ref_l
        del beam_gridded_l
        del beam_gridded_offset_l
        ############################################

        # Remove bad KIDs from goodlist (== total KID list)
        betterlist = np.delete(CDP["goodlist"], badlist)

        # Remove wideband KIDs
        betterlist_nowb = np.delete(CDP["goodlist"], wideband)
        betterlist_nowb = np.delete(betterlist_nowb, badlist)
        if plot_beams:
            # Plot amp and phase beam
            tonelabels = [x/1e9 for x in f_RF]
            clog.info("Plotting all beam patterns. This may take a while")
            for k in tqdm(range(0, len(XYT["ref_range"])), ncols=100, colour="green"):
                if k != 3 and k != 12:
                    continue
                os.system("mkdir images{}/beams/{}".format(dirmeas, k))
                for j in CDP["goodlist"]:
                    fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})

                    divider1 = make_axes_locatable(ax[0])
                    divider2 = make_axes_locatable(ax[1])

                    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                    
                    grid_x, grid_y = np.mgrid[x_uniq[0]:x_uniq[-1]:x_uniq.shape[0] * 1j,
                                            y_uniq[0]:y_uniq[-1]:y_uniq.shape[0] * 1j]

                    maxplot = np.max(20*np.log10(np.absolute((beam_corrected_l[j])[k,:,:])/wideband_norm[k]))

                    ampfig = ax[0].pcolormesh(grid_x, grid_y, 20*np.log10(np.absolute((beam_corrected_l[j])[k,:,:].T)/wideband_norm[k]), vmin=maxplot-30, vmax=maxplot)
                    phasefig = ax[1].pcolormesh(grid_x, grid_y, np.angle((beam_corrected_l[j])[k,:,:].T))
                    ax[0].set_title("WNA / [dB]", y=1.08)
                    ax[0].set_box_aspect(1)
                    ax[0].set_xlabel(r"$x$ / [mm]")
                    ax[0].set_ylabel(r"$y$ / [mm]")
                    ax[1].set_title("Phase / [rad]", y=1.08)
                    ax[1].set_xlabel(r"$x$ / [mm]")
                    ax[1].set_ylabel(r"$y$ / [mm]")
                    ax[1].set_box_aspect(1)

                    c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
                    c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')

                    c1.update_ticks()
                    c2.update_ticks()

                    pt.suptitle("KID {}, reftone = {:.2f} GHz".format(j, tonelabels[k]))
                    pt.savefig(fname="images" + dirmeas + "/beams/{}/KID_{}.png".format(k,j),bbox_inches='tight', dpi=300)
                    pt.cla()
                    pt.clf()
                    pt.close()

    # Find max KID for each ref tone.
    max_ind = np.zeros(len(XYT["ref_range"]), dtype=int)
    max_ind_Idx = np.arange(0, len(max_ind))
    for i in range(len(XYT["ref_range"])):
        # without wideband
        max_ind[i] = np.argmax(np.absolute(max_beam_l[betterlist,i]))
        max_ind[i] = betterlist[max_ind[i].astype(int)].astype(int)

    max_ind = np.array(max_ind) # KID index
    #max_ind[3] = 117
    max_ind_Idx = np.array(max_ind_Idx) # Reftone index
    
    #chunk = np.load("temp/chunks/chunk_5.npy")
    chunk = np.load("temp/xy/xy_5.npy")

    # Visual KID vs harmonic number
    if plot_beams_max:
        # Plot amp and phase beam for loudest KIDs
        if not os.path.isdir('images{}/beams/maxbeams/'.format(dirmeas)):
            os.system("mkdir images{}/beams/maxbeams/".format(dirmeas))
        tonelabels = [x/1e9 for x in f_RF]
        KID_list_plot = []
        clog.info("Plotting maximum KID beam patterns")
        for Id, j in tqdm(enumerate(max_ind.astype(int)), ncols=100, colour="green", total=len(max_ind)):
            fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.55})

            grid_x, grid_y = np.mgrid[x_uniq[0]:x_uniq[-1]:x_uniq.shape[0] * 1j,
                                    y_uniq[0]:y_uniq[-1]:y_uniq.shape[0] * 1j]

            max_max = np.max(np.absolute((beam_corrected_l[j])[Id,:,:]))
            maxplot = np.max(20*np.log10(np.absolute((beam_corrected_l[j])[Id,:,:])/wideband_norm[Id]))
            to_save = np.flip((beam_corrected_l[j])[Id,:,:], axis=-1)
            
            ampfig = ax[0].pcolormesh(grid_x, grid_y, 20*np.log10(np.absolute((beam_corrected_l[j])[Id,:,:].T)/max_max), 
                                    cmap=cmocean.cm.thermal, vmin=-30, vmax=0, linewidth=0, rasterized=True)
            phasefig = ax[1].pcolormesh(grid_x, grid_y, np.angle((beam_corrected_l[j])[Id,:,:].T), cmap=cmocean.cm.phase, linewidth=0, rasterized=True)
            ax[0].set_title("Power / dB")#, y=1.08)
            ax[0].set_xlabel(r"($x-x_\mathrm{WF}$) / mm")
            ax[0].set_ylabel(r"($z-z_\mathrm{WF}$) / mm")
            ax[1].set_title("Phase / rad")#, y=1.08)
            ax[1].set_xlabel(r"($x-x_\mathrm{WF}$) / mm")
            ax[1].set_ylabel(r"($z-z_\mathrm{WF}$) / mm")

            # Plot a cross through center
            divider1 = make_axes_locatable(ax[0])
            divider2 = make_axes_locatable(ax[1])

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)

            phasefig.set_clim(-np.pi, np.pi)

            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')#,fraction=0.046)
            c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])#,fraction=0.046)
            
            c2.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$',' 0', r'$\pi/2$', r'$\pi$'])  # vertically oriented colorbar
            
            #c1.update_ticks()
            #c2.update_ticks()
            ax[0].set_aspect(1)
            ax[1].set_aspect(1)

            #pt.suptitle("KID {}, $f$ = {:.2f} GHz".format(j, tonelabels[Id]))
            pt.savefig(fname="images" + dirmeas + "/beams/maxbeams/ref_{}.pdf".format(Id),bbox_inches='tight', dpi=300)
            pt.cla()
            pt.clf()
            pt.close()

            KID_list_plot.append([int(j), tonelabels[Id]])

        np.savetxt(f"./{dirmeas}_KID_lists.txt", KID_list_plot, fmt=('%3i %.3f'))

    if find_center:

        x_center_beam = []
        y_center_beam = []
        for harmIdx, KIDIdx in enumerate(max_ind):
            beam = beam_corrected_l[KIDIdx][harmIdx,:,:]

            xc, yc = fc.calcCenter(x_gridded, y_gridded, np.absolute(beam))

            x_center_beam.append(xc)
            y_center_beam.append(yc)
        
        x_center_beam = np.array(x_center_beam)
        y_center_beam = np.array(y_center_beam)

        m_xc = np.mean(x_center_beam)
        m_yc = np.mean(y_center_beam)
        cov_xc = np.std(x_center_beam)
        cov_yc = np.std(y_center_beam)

        clog.info(" *** MEASURED BEAM CENTERS, MP COORDS ***")
        clog.info("X = {:.2f} +- {:.2f} mm".format(m_xc, cov_xc))
        clog.info("Y = {:.2f} +- {:.2f} mm".format(m_yc, cov_yc))


    ############################## GAUSSIAN BEAM FITTING ###############################################
    FPG = st.FPG
    '''
    FPG["w0_x"]         = 4
    FPG["w0_y"]         = 4
    FPG["dzxy"]         = 2
    FPG["xi_gb"]        = y_center
    FPG["yi_gb"]        = x_center
    FPG["zi_gb"]        = -370
    FPG["theta_x0"]     = 1e-2
    FPG["theta_y0"]     = -5e-2
    FPG["theta_z0"]     = 1e-2
    '''
    FPG["w0_x"]         = 4
    FPG["w0_y"]         = 4
    FPG["dzxy"]         = 2
    FPG["xi_gb"]        = x_center
    FPG["yi_gb"]        = y_center
    FPG["zi_gb"]        = -250
    FPG["theta_x0"]     = -1e-2
    FPG["theta_y0"]     = -5e-2
    FPG["theta_z0"]     = 1e-2
    #'''
    if fit_gauss:
        eff_l = []
        optargs_l = []
        for harmIdx, KIDIdx in enumerate(max_ind):
            #if harmIdx == 3 or harmIdx == 13:
            #    continue
            beam = {}
            beam["lambda"]      = 2.998e11 / f_RF[harmIdx]
            beam["cdata"]       = beam_corrected_l[KIDIdx][harmIdx,:,:]
            beam["x"]           = x_gridded
            beam["y"]           = y_gridded

            ############################# CALL TO NESTED_ASTIGGAUSSIAN_FIT ############################
            temp_optargs, temp_eff = nested_astiggausian_fit(FPG,
                                                             beam,
                                                             dirmeas,
                                                             harmIdx,
                                                             KIDIdx,
                                                             plot_fit_flag)
            ###########################################################################################
            
            if temp_eff['coupling'] < 0.6:
                clog.warning(f"Gaussicity @ {f_RF[harmIdx]/1e9:.3f} GHz : {temp_eff['coupling']:.3f}")

            else:
                clog.info(f"Gaussicity @ {f_RF[harmIdx]/1e9:.3f} GHz : {temp_eff['coupling']:.3f}")
            eff_l.append(temp_eff)
            optargs_l.append(temp_optargs)

        filehandler = open("temp/gauss_fit/eff_l","wb")
        pickle.dump(eff_l,filehandler)
        filehandler.close()
        filehandler = open("temp/gauss_fit/optargs_l","wb")
        pickle.dump(optargs_l,filehandler)
        filehandler.close()

    else:
        filep = open("temp/gauss_fit/eff_l",'rb')
        eff_l = pickle.load(filep)
        filep.close()
        filep = open("temp/gauss_fit/optargs_l",'rb')
        optargs_l = pickle.load(filep)
        filep.close()
   
    check = np.array([x["coupling"] for x in eff_l])
    idx_good = np.squeeze(np.argwhere(check > 0.5))

    mtilt_x = np.degrees(np.mean(np.array([x["theta_x"] for x in eff_l])[idx_good]))
    mtilt_y = np.degrees(np.mean(np.array([x["theta_y"] for x in eff_l])[idx_good]))
    mtilt_z = np.degrees(np.mean(np.array([x["theta_z"] for x in eff_l])[idx_good]))
    cov_x = np.degrees(np.std(np.array([x["theta_x"] for x in eff_l])[idx_good]))
    cov_y = np.degrees(np.std(np.array([x["theta_y"] for x in eff_l])[idx_good]))
    cov_z = np.degrees(np.std(np.array([x["theta_z"] for x in eff_l])[idx_good]))

    clog.info(" *** BEAM TILTS GAUSS, MP COORDS ***")
    clog.info("X = {:.2f} +- {:.2f} deg".format(-mtilt_x, cov_x))
    clog.info("Y = {:.2f} +- {:.2f} deg".format(-mtilt_y, cov_y))
    clog.info("Z = {:.2f} +- {:.2f} deg".format(-mtilt_z, cov_z))

    dx = np.mean(np.array([x["xi_gp"] - x_center for x in eff_l])[idx_good])
    dy = np.mean(np.array([x["yi_gp"] - y_center for x in eff_l])[idx_good])
    dz = np.mean(np.array([x["zi_gp"] for x in eff_l])[idx_good])
    cov_dx = np.std(np.array([x["xi_gp"] - x_center for x in eff_l])[idx_good])
    cov_dy = np.std(np.array([x["yi_gp"] - y_center for x in eff_l])[idx_good])
    cov_dz = np.std(np.array([x["zi_gp"] for x in eff_l])[idx_good])

    clog.info(" *** FOCUS OFFSETS GAUSS W.R.T. MP CENTER ***")
    clog.info("X = {:.2f} +- {:.2f} mm".format(dx, cov_dx))
    clog.info("Y = {:.2f} +- {:.2f} mm".format(dy, cov_dy))
    clog.info("Z = {:.2f} +- {:.2f} mm".format(dz, cov_dz))

    #print(x_center_beam, y_center_beam)
    #print(dx, cov_dx)
    #print(dy, cov_dy)
    #print(dz, cov_dz)

    grid_params = np.array([[x_center, y_center], [len(x_uniq), len(y_uniq)], [min(x_uniq), max(x_uniq)], [min(y_uniq), max(y_uniq)]])
    beam_params = np.array([[np.mean(x_center_beam), np.mean(y_center_beam)], [dx, cov_dx], [dy, cov_dy], [dz, cov_dz]])
    tilt_params = np.array([[-mtilt_x, cov_x], [-mtilt_y, cov_y], [-mtilt_z, cov_z]])

    if plot_gauss_param:
        xr = f_RF[max_ind_Idx]/1e9
        selector = lambda x : [effc[x] for effc in eff_l]
        fig, ax = pt.subplots(2,2, figsize=(10,10), gridspec_kw={'wspace':0.3,'hspace':0.2})

        w0x = selector("w0x")
        w0y = selector("w0y")

        w0m = (np.array(w0x) + np.array(w0y)) / 2

        np.savetxt("w0m.txt", w0m)

        ax[0][0].plot(xr, selector("coupling"), color='black', linewidth=0.5,zorder=0)
        ax[0][0].scatter(xr, selector("coupling"), color='blue',zorder=1)
        ax[0][0].set_xlabel(r"$f$ / [GHz]")
        ax[0][0].set_ylabel(r"Gaussicity")
        ax[0][0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[0][0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[0][0].set_box_aspect(1)

        ax[0][1].plot(xr, selector("w0x"), color='black', linewidth=0.5,zorder=0)
        ax[0][1].plot(xr, selector("w0y"), color='black', linewidth=0.5,zorder=0)
        ax[0][1].scatter(xr, selector("w0x"), color='blue',zorder=1, label=r"$w_{0,x}$")
        ax[0][1].scatter(xr, selector("w0y"), color='red',zorder=1, label=r"$w_{0,y}$")
        ax[0][1].set_xlabel(r"$f$ / [GHz]")
        ax[0][1].set_ylabel(r"Beamwaist / [mm]")
        ax[0][1].legend(frameon=False, prop={'size': 20}, loc='upper right',handlelength=1)
        ax[0][1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[0][1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[0][1].set_box_aspect(1)

        ax[1][0].plot(xr, selector("zi_gp"), color='black', linewidth=0.5,zorder=0)
        ax[1][0].plot(xr, [x+y for x,y in zip(selector("zi_gp"),selector("dzxy"))], color='black', linewidth=0.5,zorder=0)
        ax[1][0].scatter(xr, selector("zi_gp"), color='blue',zorder=1, label=r"$z_0$")
        ax[1][0].scatter(xr, [x+y for x,y in zip(selector("zi_gp"),selector("dzxy"))], color='red',zorder=1, label=r"$z_0+d_{zxy}$")
        ax[1][0].set_xlabel(r"$f$ / [GHz]")
        ax[1][0].set_ylabel(r"Beamwaist position / [mm]")
        ax[1][0].legend(frameon=False, prop={'size': 20}, loc='upper right',handlelength=1)
        ax[1][0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[1][0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[1][0].set_box_aspect(1)

        ax[1][1].plot(xr, [360/(2*np.pi) * x for x in selector("theta_x")], color='black', linewidth=0.5,zorder=0)
        ax[1][1].plot(xr, [360/(2*np.pi) * x for x in selector("theta_y")], color='black', linewidth=0.5,zorder=0)
        ax[1][1].scatter(xr, [360/(2*np.pi) * x for x in selector("theta_x")], color='blue',zorder=1,label=r"$\theta_y$")
        ax[1][1].scatter(xr, [360/(2*np.pi) * x for x in selector("theta_y")], color='red',zorder=1,label=r"$\theta_x$")
        ax[1][1].set_xlabel(r"$f$ / [GHz]")
        ax[1][1].set_ylabel(r"Beam tilt / [deg]")
        ax[1][1].legend(frameon=False, prop={'size': 20},handlelength=1)
        ax[1][1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[1][1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax[1][1].set_box_aspect(1)

        pt.savefig(fname="images" + dirmeas + "/beams/gauss/fitparam.pdf",bbox_inches='tight', dpi=300)
        #pt.show()
        pt.close()

        fig, ax = pt.subplots(1,1, figsize=(5,5))
        ax.plot(xr, [360/(2*np.pi) * x for x in selector("theta_z")], color='black', linewidth=0.5,zorder=0)
        ax.scatter(xr, [360/(2*np.pi) * x for x in selector("theta_z")], color='purple',zorder=1)
        ax.set_ylabel(r"Beam tilt / [deg]")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_box_aspect(1)
        pt.savefig(fname="images" + dirmeas + "/beams/gauss/theta_z.pdf",bbox_inches='tight', dpi=300)
        #pt.show()
        pt.close()

    clog.info(f"Saving maximum beam patterns, Gaussian fit results and grid parameters to analysis_results{dirmeas}")
    results_path = 'analysis_results{}/'.format(dirmeas) 
    if not os.path.isdir(results_path):
        os.makedirs("analysis_results{}/".format(dirmeas))
   
    np.save(results_path + "grid_params", grid_params)
    np.save(results_path + "beam_params", beam_params)
    np.save(results_path + "tilt_params", tilt_params)
    np.save(results_path + "gaussicity", check)
    np.save(results_path + "used_KIDs", max_ind)

    for Id, j in enumerate(max_ind.astype(int)):
        to_save = np.flip((beam_corrected_l[j])[Id,:,:], axis=-1)

        np.savetxt(results_path + "r{}.txt".format(round(tonelabels[Id])), np.real(to_save))
        np.savetxt(results_path + "i{}.txt".format(round(tonelabels[Id])), np.imag(to_save))
    
    clog.info("FINISHED. ELAPSED TIME = {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    Main()
