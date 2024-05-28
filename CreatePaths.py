import os

def CreatePaths(dirmeas):
    moviepath = 'images{}/movie'.format(dirmeas)

    exists_temp = os.path.isdir('temp/')
    exists_mbeam = os.path.isdir('max_beam/')
    exists_mbeamfoc = os.path.isdir('meas_beam_focus/')
    exists_beamdir = os.path.isdir('images{}/beams'.format(dirmeas))
    exists_circle_fits = os.path.isdir('images{}/circle_fits'.format(dirmeas))
    exists_maxbeamdir = os.path.isdir('images{}/beams/maxbeams'.format(dirmeas))
    exists_maxbeamfft = os.path.isdir('images{}/beams/maxbeams/fft'.format(dirmeas))
    exists_gauss = os.path.isdir('images{}/beams/gauss'.format(dirmeas))
    exists_gauss_slice = os.path.isdir('images{}/beams/gauss/slices'.format(dirmeas))
    exists_gauss_far_field = os.path.isdir('images{}/beams/gauss/far_field'.format(dirmeas))

    exists_movie = os.path.isdir(moviepath)

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

    if not exists_mbeam:
        os.makedirs('max_beam')

    if not exists_mbeamfoc:
        os.makedirs('meas_beam_focus')

    if not exists_beamdir:
        os.makedirs('images{}/beams'.format(dirmeas))

    if not exists_circle_fits:
        os.makedirs('images{}/circle_fits'.format(dirmeas))

    if not exists_maxbeamdir:
        os.makedirs('images{}/beams/maxbeams'.format(dirmeas))

    if not exists_maxbeamfft:
        os.makedirs('images{}/beams/maxbeams/fft'.format(dirmeas))

    if not exists_gauss:
        os.makedirs('images{}/beams/gauss'.format(dirmeas))

    if not exists_gauss_slice:
        os.makedirs('images{}/beams/gauss/slices'.format(dirmeas))

    if not exists_gauss_far_field:
        os.makedirs('images{}/beams/gauss/far_field'.format(dirmeas))

    if not exists_movie:
        os.makedirs('images{}/movie'.format(dirmeas))

def CreatePathsHex(dirmeas):
    moviepath = 'images{}/movie'.format(dirmeas)

    exists_temp = os.path.isdir('hex_data/temp/')
    exists_npy = os.path.isdir(f'hex_data/{dirmeas}/')
    exists_images = os.path.isdir(f'hex_data/images{dirmeas}/cold_spot/')

    if not exists_temp:
        os.makedirs('hex_data/temp/chunks/')
        os.makedirs('hex_data/temp/x/')
        os.makedirs('hex_data/temp/y/')
        os.makedirs('hex_data/temp/z/')
        os.makedirs('hex_data/temp/xy/')
        os.makedirs('hex_data/temp/KDP/')

    if not exists_npy:
        os.makedirs(f'hex_data/{dirmeas}/')

    if not exists_images:
        os.makedirs(f'hex_data/images{dirmeas}/cold_spot/')

