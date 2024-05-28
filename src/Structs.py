
################################### STRUCT LIST #######################################
# This list is here to provide an overview of structures often encountered throughout #
# the main scripts. Each structure is stated with corresponding members.              #
#######################################################################################

# Initializing xy_settings
xy_settings = {
    "ref_multiplier"    :32,        # reference harmonic
    "ref_range"         :[],        # Range of harmonics        
    "xy_trace"          :[],        # Grid information
    "ref_power"         :1,         #
    "f_chopper"         :13,        # Chopper frequency
    "end_sett"          :0,         # Cut from end points
    "start_sett"        :0,         # Cut from start
    "subchunks"         :1,         #
    "nrtr"              :480,       # Nr. of datapoints per xy point
    "nr_in_memory"      :4800,      # 
    "xtr_range"         :0,         #
    "filter_FWHM"       :0,         #
    "range_z"           :-60,       # z-range, I guess?
    "deglitch"          :0,         #
    "guesswidth"        :10,        # Estimate of beamwidth
    "cut"               :0.05,      # Fit above 10% of peak
    "meas"              :"f_resp",  #
    "trace_mult"        :4,         #
    "set_goodlist"      :"1st pt",  #
    "peakpos_ref"       :6,         #
    "peakpos"           :161,       # 
    "ref_set"           :"abs",     #
    "nrpt"              :20504,     # Nr. of points per row of a scan
    "mod_frac"          :5          # Fraction of DAQ rate
}

# Initializing CARDparam
CARDparam = {
    "dumptime"          :7.8643e-4, #
    "nbins"             :65536,     #
    "CARDBW"            :2e9,       #
    "smooth_wdow_size"  :20,        #
    "nr_refs"           :0,         #
    "nr_blinds"         :100,       #
    "nr_kids"           :11,        #
    "tonenumber"        :111,       #
    "LO"                :4.78e9,    #
    "KIDbins"           :[],        #
    "blindbins"         :[],        #
    "refbins"           :[],        #
    "KID_p"             :[],        #
    "blind_p"           :[],        #
    "p"                 :-70.036,   #
    "goodlist"          :[],        #
    "blindlist"         :[],        #
    "ref_tones"         :[],        #
    "refKID"            :1,         #
    "scanwidth"         :6e6,       #
    "datatype"          :0,         #
    "binwidth"          :0          #
}

CALparam = {
    "tone"              :[],        #
    "Quadrant"          :[],        #
    "sweepfile"         :[],        # File with readout frequencies
    "phaseref"          :[],        #
    "S21max"            :[],        #
    "KIDcalphase"       :[],        #
    "KIDcalreal"        :[],        #
    "KIDcalR"           :[],        #
    "xfit"              :[],        #
    "yfit"              :[]         #
}

data = {
    "comdatasweep"      :[],        #
    "f"                 :[],        #
    "normdatasweep"     :[],        #
    "KIDcalS21data"     :[]         #
}

KIDparam = {
    "abs_value"         :[],        #
    "theta0"            :[],        #
    "phi0"              :[],        #
    "fres"              :[],        #
    "S21min"            :[],        #
    "Q"                 :[],        #
    "Qi"                :[],        #
    "Qc"                :[],        #
    "Qr"                :[],        #
    "f_used"            :[],        #
    "KIDBW"             :[],        #
    "diffed"            :[],        #
    "dthetadfmin"       :[],        #
    "dthetadf_used"     :[],        #
    "f_used"            :[],        #
    "S21_used"          :[],        #
    "poly"              :[]         #
}

ref = {
    "f_fft"             :[],        #
    "peakpos_ref"       :[],        #
    "peak_f_ref"        :[],        #
    "peakpos"           :[],        #
    "peak_f"            :[],        #
    "fft"               :[],        #
    "raw"               :[],        #
    "KIDcalibrated"     :[],        #
    "comdata"           :[],        #
    "normdata"          :[],        #
    "reference_scan"    :[],        #
    "reference_fft"     :[]         #
}

# fit_param_gauss
FPG = {
    "w0_x"              :[],        # Beamwaist in x-direction at z0
    "w0_y"              :[],        # Beamwaist in y-direction at z0
    "dzxy"              :[],        # 
    "xi_gb"             :[],        # 
    "yi_gb"             :[],        #
    "zi_gb"             :[],        #
    "theta_x0"          :[],        # Opening angle in x-direction
    "theta_y0"          :[],        # Opening angle in y-direction
    "theta_z0"          :[]         # 
}

beam = {
    "lambda"            :[],        #
    "cdata"             :[],        #
    "x"                 :[],        #
    "y"                 :[]         #
}

# efficiencies_gauss
EG = {
    "coupling"          :[],        #
    "w0x"               :[],        #
    "w0y"               :[],        #
    "dzxy"              :[],        #
    "xi_gp"             :[],        #
    "yi_gp"             :[],        #
    "zi_gp"             :[],        #
    "theta_x"           :[],        #
    "theta_y"           :[],        #
    "theta_z"           :[]
}

meas_dat = {
    "lambda"            :[],        #
    "Xmp"               :[],        #
    "Ymp"               :[],        #
    "Zmp"               :[],        #
    "cpx"               :[]         #
}

GBA_dat = {
    "X"                 :[],        #
    "Y"                 :[],        #
    "Z"                 :[],        #
    "zy"                :[],        #
    "zcx"               :[],        #
    "zcy"               :[],        #
    "wx"                :[],        #
    "wy"                :[],        #
    "R_x"               :[],        #
    "R_y"               :[],        #
    "phiX"              :[],        #
    "phiY"              :[],        #
    "psi"               :[],        #
    "c00"               :[],        #
    "eta"               :[],        #
    "epsilon"           :[]         #
}
    
    
#### TODO: find meaning of ALL variables
