import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_beam(path, name):
    """
    Read a beam pattern from some directory.
    
    @param path Path relative to cwd where beam is.
    @param name Name of beam pattern. Should be stored in two text files in path, one prependen with 'i' (complex part), other with 'r' (real part).
    """

    get_path = lambda x : os.path.join(path, x + name + ".txt")

    try:
        beam = np.loadtxt(get_path('r')) + 1j * np.loadtxt(get_path('i'))

    except:
        print(f"Failed to load {name} from {path}!")
        return

    beam_dB = 20 * np.log10(np.absolute(beam) / np.max(np.absolute(beam)))
    beam_arg = np.angle(beam)

    grid_params = np.load(os.path.join(path, "grid_params.npy"))
    nx, ny = grid_params[1]

    xmin, xmax = grid_params[2]
    ymin, ymax = grid_params[3]

    extent = [xmin, xmax, ymin, ymax]

    fig, ax = plt.subplots(1,2)

    div_dB = make_axes_locatable(ax[0])
    cax_dB = div_dB.append_axes('right', size='5%', pad=0.05)

    div_arg = make_axes_locatable(ax[1])
    cax_arg = div_arg.append_axes('right', size='5%', pad=0.05)

    dB_fig = ax[0].imshow(beam_dB, origin="lower", extent=extent)
    arg_fig = ax[1].imshow(beam_arg, origin="lower", extent=extent)
    
    fig.colorbar(dB_fig, cax=cax_dB, orientation='vertical')
    fig.colorbar(arg_fig, cax=cax_arg, orientation='vertical')

    plt.show()

if __name__ == "__main__":
    path = "./analysis_results/XYZScan_20230111_194931"
    name = "205"
    read_beam(path, name)

