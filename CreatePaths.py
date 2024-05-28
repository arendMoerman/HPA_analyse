import os

def CreatePaths(dirmeas):
    moviepath = 'images{}/movie'.format(dirmeas)

    exists_temp = os.path.isdir('temp/')
    exists_beamdir = os.path.isdir('images{}/beams'.format(dirmeas))
    exists_maxbeamdir = os.path.isdir('images{}/beams/maxbeams'.format(dirmeas))
    exists_gauss = os.path.isdir('images{}/beams/gauss'.format(dirmeas))

    if not exists_temp:
        os.makedirs('temp/chunks/')
        os.makedirs('temp/x/')
        os.makedirs('temp/y/')
        os.makedirs('temp/z/')
        os.makedirs('temp/xy/')
        os.makedirs('temp/gauss_fit/')
        os.makedirs('temp/beams/')
        os.makedirs('temp/beam_ref/')
        os.makedirs('temp/blindcors/')
        os.makedirs('temp/KDP/')

    if not exists_beamdir:
        os.makedirs('images{}/beams'.format(dirmeas))

    if not exists_maxbeamdir:
        os.makedirs('images{}/beams/maxbeams'.format(dirmeas))

    if not exists_gauss:
        os.makedirs('images{}/beams/gauss'.format(dirmeas))
