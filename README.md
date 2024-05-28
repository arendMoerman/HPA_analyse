# Analyse HPA measurements
Analyse and plot HPA beam pattern measurements.
Prereqs: `numpy`, `scipy`, `astropy`, `matplotlib`, `tqdm`, `cmocean`

## Step one
In `BeamAnalysis.py`, navigate to line 64. From here, there are a few control toggles. They can be turned off by setting to 0, or on by setting to 1.
`chopFlag` : Chop large .FITS file into smaller chunks and store in temp/ folder.
`readData` : Read the data, where one CPU core reads one chunk.
NOTE: if, for a given measurement file, a run has been performed with `chopFlag = 1` and `readData = 1`, for subsequent analyses they can be set to 0, saving time. This is because all the results are stored in the temp/ folder, which the program will find. When another measurement is to be analysed, do not forget to put these toggles back to 1.
`Grid_data` : Assign TOD data onto a 2D grid of the HPA scanner. Best to leave at 1.
`find_center` : Finds amplitude center of beam patterns and logs to screen.
`fit_gauss` : Fit a Gaussian beam to measured beam.
`plot_beams` : Plot beams for all KIDs, for all HPA frequencies, mostly for troubleshooting. WARNING: only run if you want ALL plots, can take a while.
`plot_beams_max` : Plot the beam pattern of the KID that has the largest response for a given HPA frequency.
`plot_fit_flag` : Plot beam patterns and Gaussian fit as contourplot.
`plot_gauss_param` : Plot the fitted parameters for each HPA frequency.

## Step two
Edit `BeamAnalysis.py` so that `dirmeas` and `path` variables point to the measurement folder.
`dirmeas` (line 85) should be a string containing name of measurement and `path` (line 86) the path to measurement.

Example: I want to analyse XYZScan\_20230111\_194931, located in /home/arend/Projects/HPA/measurements:
`dirmeas = "/XYZScan_20230111_194931"`
`path = "/home/arend/Projects/HPA/measurements"`

## Step three
Make sure that you use an appropriate KIDid to filter frequency conversion file. I added KIDID\_FilterF0\_Q.run991.csv file.

## Running the script
```
python BeamAnalysis.py 
```

Please note that a couple of folders and files will be created, so run this from a location where you dont mind clogging too much.
Also, the script might take a while to analyse and process (large) .FITS files, so be patient.

The plots (if specified) are in images/{name\_of\_measurement}/beams/
Also, a file named {name\_of\_measurment}\_KID\_lists.txt is created, which contains the KIDid and HPA frequency of the KID.

The `analysis_results` folder contains beam patterns and several .npy files for further analysis.
`grid_params.npy` : Contains an array containing tuples with scanning plane parameters, which in turn contain:
x center coordinate, y center coordinate (in measurement plane coordinates, mm)
number of x points, number of y points
smallest x value, max x value (mm)
smallest y value, max y value (mm)

`beam_params.npy` : Contains array with fitted beam parameter tuples:
mean (across all HPA frequencies) x coordinate, mean y coordinate of center of beam, in measurement plane coordinates, mm
focal offset of fitted Gaussian beam along x axis w.r.t. beam center, standard error
focal offset of fitted Gaussian beam along y axis w.r.t. beam center, standard error
focal offset of fitted Gaussian beam along z axis w.r.t. scanning plane surface, standard error
All these parameters are in millimeters.

`gaussicity.npy` contains the Gaussicity of each fit to each HPA frequency.

The beam patterns are saved in a text file format with the name being the HPA frequency in GHz, with the complex (prefix 'i') and real parts (prefix 'r') in separate files. Also, the `used_KIDs.npy` file contains the same as the {name\_of\_measurment}\_KID\_lists.txt (will probably remove in the future). The `tilt_params.npy` file contains the fitted tilt of the beam, for each HPA frequency, in degrees around the lateral axes in the measurement plane.
