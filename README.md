# Analyse HPA measurements
Analyse and plot HPA beam pattern measurements.
Prereqs: numpy, scipy, astropy, matplotlib, tqdm

## Step one
Edit ```BeamAnalysis.py``` so that `dirmeas` and `path` variables point to the measurement folder.
`dirmeas` (line 85) should be a string containing name of measurement and `path` (line 86) the path to measurement.

Example: I want to analyse XYZScan_20230111_194931, located in /home/arend/Projects/HPA/measurements:
```dirmeas = "/XYZScan_20230111_194931"```
```path = "/home/arend/Projects/HPA/measurements"```

## Step two
Make sure that you use an appropriate KIDid to filter frequency conversion file. I added KIDID_FilterF0_Q.run991.csv file.

## Running the script
```
python BeamAnalysis.py 
```
