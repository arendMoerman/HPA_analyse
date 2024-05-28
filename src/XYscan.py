import math
import cmath
import numpy as np
import sys
import time
import src.Utils as ut
import src.Structs as st
from src.BlindCorrection import BlindCorrection
from src.KIDcal import KIDcal
from scipy.fft import fft, ifft
import scipy.fft as ft
import scipy.optimize as opt
import pickle

from tqdm import tqdm

from astropy.io import fits
import matplotlib.pyplot as pt
import matplotlib.ticker as ticker
from matplotlib import cm
   
from multiprocessing import Pool
from functools import partial

def Theta_to_freq(theta, theta0, Q):
    """
    Converts phase response to frequency shift.
    - theta: [nbtr, goodlist] sized array of floats containing phase responses.
    - theta0: [goodlist] sized array of floats containing KID phase offsets.
    - Q: [goodlist] sized array of floats containing Q-factors at resonance
    """
    
    return -np.tan((theta + theta0)/2) / (2*Q)

def ChopRange(args, rawfilename, XYT):
    """
    Subdivides .FITS file into smaller chunks, according to number of cores used.
    Saves binary numpy tables. Zip args before call.
    args ->
    - num_cores: int number of CPU cores to write chunks for.
    - chunks: Array of size +- [int(nrpt/num_cores)] containing indices of good 
      positions in .FITS file.
    const ->
    - rawfilename: path, including name, of .FITS file.
    """
    
    num_cores, chunks = args
    
    if type(rawfilename) != str:
        raise ValueError("Path to .FITS file should be a string.")
    
    with fits.open(rawfilename, memmap=True) as table:
        cols = table[1].data.field('data') # Obtain all data this way

        filelist = []
        
        filelist.append("temp/chunks/chunk_{}.npy".format(num_cores))
        chunk_end = chunks + XYT["nrtr"]
            
        ChopRange = np.arange(chunks[0], chunk_end[0])
  
        for j in range(len(chunks)-1): # First iteration is above, to initialize ChopRange
            ChopRange = np.concatenate((ChopRange, np.arange(chunks[j+1], chunk_end[j+1])))

        np.save("temp/chunks/chunk_{}".format(num_cores), cols[ChopRange])#, newline=" ", fmt='%i')
            
    return filelist
            
def ProcessFITS(args,
                fft_gain, 
                CALparam,
                CDP,
                XYT,
                KDP,
                goodlist,
                num_cores,
                toneIdx,
                offsets,
                dirmeas, clog):
    """
    Read chunks obtained from .FITS file. Each chunk is sent to separate
    core for optimal processing.
    """
    
    chunk, it = args
    S21 = np.zeros((XYT["nbtr"], CDP["tonenumber"]), dtype=complex)
    tonelist = np.arange(CDP["tonenumber"])
    megaArr = np.zeros((XYT["nbtr"], 2*len(tonelist)))
    
    table = np.load("temp/chunks/chunk_{}.npy".format(it), mmap_mode='r')
    
    # Initialize xy data structure, can be joined together later for data_cube
    xy = np.zeros((len(XYT["ref_range"]), len(goodlist), len(chunk)), dtype=complex)
    
    # Calculate corresponding xyz range
    coordrange = np.arange(offsets[it], offsets[it+1]).astype(int)
    
    x = XYT["xy_trace"][coordrange,0]
    y = XYT["xy_trace"][coordrange,1]
    z = XYT["xy_trace"][coordrange,4]
    beam_ref = XYT["xy_trace"][coordrange,5]
    
    np.save("temp/x/x_{}".format(it), x)
    np.save("temp/y/y_{}".format(it), y)
    np.save("temp/z/z_{}".format(it), z)
    np.save("temp/beam_ref/beam_ref_{}".format(it), beam_ref)
    
    #gridinfo = np.array([x.shape, y.shape, np.min(x), np.max(x), np.min(y), np.max(y)])
    
    #np.savetxt("meas_beam_focus/gridinfo.txt", gridinfo)
    
    blindcor_mat = []
    
    ######################### Stuff for checking blindcorr intervals ########################
    std_blindcorr_neg = np.zeros(len(chunk))
    std_blindcorr_pos = np.zeros(len(chunk))
    #########################################################################################
    if it == 0:
        iterable = tqdm(range(len(chunk)), ncols=100, colour="green", total=len(chunk))

    else:
        iterable = range(len(chunk))
    
    for j in iterable:
        end = XYT["nbtr"]
        megaArr[:,2*tonelist] = np.array(table[j*XYT["nbtr"]:j*XYT["nbtr"]+end])[:,2*toneIdx] / fft_gain
        megaArr[:,2*tonelist+1] = np.array(table[j*XYT["nbtr"]:j*XYT["nbtr"]+end])[:,2*toneIdx+1] / fft_gain
        
        S21[:,tonelist] = (megaArr[:,2*tonelist] + 1j*megaArr[:,2*tonelist+1])
        
        comdata = S21[:,tonelist] # Uncalibrated data for reference extraction
        
        if not j % 100:
            #if it == 0:
                #print("# Progress: {} / {}".format(j*num_cores,XYT["nrpt"]))
                
            normdata, KID_list_usen, KID_list_usep, zi_neg, zi_pos = BlindCorrection(S21 * np.exp(-1j*CALparam["phaseref"]), CALparam["tone"], CDP["blindlist"], CDP["smooth_wdow_size"],1,0)
            
            normdata = normdata / CALparam["S21max"]
            
            blindcor_mat.append(zi_neg)
            blindcor_mat.append(zi_pos)
            
            std_blindcorr_neg[j] = np.std(zi_neg - np.mean(zi_neg))
            std_blindcorr_pos[j] = np.std(zi_pos - np.mean(zi_pos))
            
        else:
            S21 = S21 * np.exp(-1j*CALparam["phaseref"])
            
            normdata[:,KID_list_usen] = S21[:,KID_list_usen] / blindcor_mat[-2]
            normdata[:,KID_list_usep] = S21[:,KID_list_usep] / blindcor_mat[-1]
            
            
            normdata = normdata / CALparam["S21max"]
        
        KIDcalibrated = KIDcal(normdata, CDP["goodlist"], CALparam, 1)
        
        if XYT["meas"] == "f_resp":
            func_inp = -np.unwrap(np.angle(KIDcalibrated * np.exp(-1j*np.pi)), axis=0)
            fresp   = np.zeros(KIDcalibrated.shape)
            fresp   = Theta_to_freq(func_inp, KDP["theta0"], KDP["Q"])

            scan = np.zeros((fresp.shape[0], CDP["nr_kids"]+1))
            scan[:,np.arange(fresp.shape[1])] = fresp
                            
            #del fresp
            
        else:
            scan = -np.unwrap(np.angle(KIDcalibrated[:,goodlist] * np.exp(-1j*np.pi)), axis=0)
        
        if CDP["nr_refs"] != 0:
            if CDP["ref_tones"][0]:
                scan[:,CDP["ref_tones"]] = []
                        
        else:
            if XYT["ref_set"] == "abs":
                scany = np.absolute(comdata[:,CDP["blindlist"]])
            else:
                pass #TODO: insert difficult expression
    
        low_sb_idx = np.squeeze(np.argwhere(CALparam["tone"][CDP["blindlist"]] < 0))
        upp_sb_idx = np.squeeze(np.argwhere(CALparam["tone"][CDP["blindlist"]] > 0))
        
        mean_low = np.mean(scany[:,low_sb_idx] - np.mean(scany[:,low_sb_idx], axis=0), axis=1)
        mean_upp = np.mean(scany[:,upp_sb_idx] - np.mean(scany[:,upp_sb_idx], axis=0), axis=1)
        
        scany = mean_low - mean_upp

        #scany = np.mean(scany - np.mean(scany, axis=0), axis=1)
        scan[:,-1] = scany # scany contains reference waveform
        
        ################################## FFT ##################################
        fftlength = int(XYT["nbtr"] / XYT["subchunks"])
        
        for n in range(XYT["subchunks"]):
            sub_pos                 = np.arange(fftlength) + n*fftlength
            sub_data                = scan[sub_pos,:]
            fftscan                 = fft(sub_data, axis=0) / fftlength
            
            sub_xy = fftscan[(XYT["peakpos"].astype(int) - 1),:]
            sub_xy[:,-1] = fftscan[(XYT["peakpos_ref"].astype(int) - 1),-1]
            
            amp = np.absolute(sub_xy[:,np.arange(sub_xy.shape[1]-1)])
            ang = np.angle(sub_xy[:,np.arange(sub_xy.shape[1]-1)])
            ang_ref = np.angle(sub_xy[0,-1])

            xy[:,:,j] = xy[:,:,j] + ((amp * np.exp(1j * ang)).T * np.exp(-1j * ang_ref * XYT["ref_range"])).T

        xy[:,:,j] = xy[:,:,j] / XYT["subchunks"]
        
        #if it == 5 and j == int(len(chunk)/2):
            #fft_chunk = ft.fftshift(fft(scan[:,171]))
            #fig, ax = pt.subplots(1,2)

            #ax[0].plot(scan[:,171])
            #np.save("scan_205GHz_middle", scan[:,171])
            #np.save("scan_215GHz_middle", scan[:,96])
            #np.save("scan_225GHz_middle", scan[:,47])
            #ax[1].plot(fft_chunk)
            #pt.show()
        
    np.save("temp/blindcors/std_blindcorr_neg_{}".format(it), std_blindcorr_neg)
    np.save("temp/blindcors/std_blindcorr_pos_{}".format(it), std_blindcorr_pos)
    
    # Save xy as .npy file and ref dict in pickle format
    np.save("temp/xy/xy_{}".format(it), xy)
    
    del xy
    
    return it


def XYscan(procdata, 
           CALparam, 
           CDP, 
           XYT, 
           datadir, 
           rawfilename, 
           KDP,
           num_cores,
           dirmeas,
           chopFlag,
           readData, clog):
    
    """
    Reads XYZ measurements from .FITS file. Called in Main.py and returns calibrated KID IQ signals.
    Inputs (for all structs, see lib/Structs.py):
    - procdata      : A data structure processed by MUXfsweep.
    - CALparam      : structure containing calibration parameters.
    - CDP           : structure containing KID information.
    - XYT           : structure containing the xy_trace.txt file, among others.
    - datadir       : path to .FITS file.
    - rawfilename   : datadir + actual .FITS name.
    - num_cores     : number of CPU cores to use for file processing
    
    No returns, instead writes processed data to 'temp' folder for further processing.
    """
    
    hdul            = fits.open(rawfilename)
    #hdul.info()
    hddata          = hdul[1].header   # Table header
    n_words         = len(hddata)
    
    goodlist        = CDP["goodlist"]
    
    fft_gain        = hddata["FFTGAIN"]
    tonenumber      = hddata["NBINS"]
    
    for i in range(n_words):
        if hddata[i] == hddata["BIN0"]:
            nbins_pt = i
            break
    
    freqpt          = np.zeros(XYT["nrpt"])
    tones           = []
    for i in range(tonenumber):
        tones.append(np.remainder(hddata[nbins_pt + i] + CDP["nbins"]/2, CDP["nbins"]) - CDP["nbins"]/2)
    
    toneIdx = []
    for i in CALparam["tone"]:
        toneIdx.append(int(np.argwhere(tones == i)))
    
    toneIdx     = np.array(toneIdx)
    tonelist    = np.arange(CDP["tonenumber"])

    nrows = hdul[1].data.shape[0]
    ndata = len(hdul[1].data[0][2])
    hdul.close()

    clog.info("Processing data using {} cores".format(num_cores))
    
    KDP["abs_value"] = np.zeros((len(goodlist),XYT["nrpt"]))
    
    ############## Write chunks to files. Select framecounts from xytrace ##############
    goodIdx = XYT["xy_trace"].T[3].astype(int) - 3 - 6 # 3 for magic, 6 for index offset
    chunks = np.array_split(goodIdx, num_cores)
    chunks = [chu for chu in chunks]
    
    offsets = np.zeros(num_cores+1)
    for i, ch in enumerate(chunks):
        offsets[i+1] = offsets[i] + len(ch)

    if chopFlag:
        ChopRangeP = partial(ChopRange, 
                             rawfilename=rawfilename,
                             XYT=XYT)
        args = zip(np.arange(num_cores), chunks)
        pool = Pool(num_cores)
        filelist = pool.map(ChopRangeP, args)
        filelist = [f[0] for f in filelist]
        clog.info("Created {} chunks for processing".format(num_cores))
        
    else:
        filelist = ["temp/chunk_{}.npy".format(i) for i in range(num_cores)]
    ###################################################################################
    
    ################## Process each chunk separately in parallel ######################
    if readData:
        ProcessFITSpar = partial(ProcessFITS,
                                 fft_gain=fft_gain, 
                                 CALparam=CALparam,
                                 CDP=CDP,
                                 XYT=XYT,
                                 KDP=KDP,
                                 goodlist=goodlist,
                                 num_cores=num_cores,
                                 toneIdx=toneIdx,
                                 offsets=offsets,
                                 dirmeas=dirmeas,
                                 clog=clog)
    
        it      = np.arange(num_cores)
    
        args    = zip(chunks, it)
        pool    = Pool(num_cores)
        result  = pool.map(ProcessFITSpar, args)
    ####################################################################################    

