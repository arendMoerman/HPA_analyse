import numpy as np
import os
from scipy import optimize
import scipy.fft as ft
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import numdifftools as nd

sec_range = 1 + 1j
APWS_sec = 1 + 1j
angle_offset = 0

def fit_sec(args):
    global sec_range
    global APWS_sec
    global angle_offset

    global beami
    global fit_param

    #cond1 = np.sqrt((beami["theta_x"] - args[3] + angle_offset)**2 + (beami["theta_y"] - args[4] + angle_offset)**2) < fit_param["sec_ang"]
    #cond2 = np.sqrt((beami["theta_x"] - args[3] + angle_offset)**2 + (beami["theta_y"] - args[4] + angle_offset)**2) > fit_param["blk_ang"]
    cond1 = np.sqrt((beami["theta_x"])**2 + (beami["theta_y"])**2) < fit_param["sec_ang"]
    cond2 = np.sqrt((beami["theta_x"])**2 + (beami["theta_y"])**2) > fit_param["blk_ang"]

    cond3 = np.absolute(beami["theta_x"] - args[3] + angle_offset) > fit_param["cross_ang"]
    cond4 = np.absolute(beami["theta_y"] - args[4] + angle_offset) > fit_param["cross_ang"]

    sec_range = cond1 & cond2# & cond3 & cond4

    APWS_sec = beami["APWS"] * np.exp(1j*(beami["kx"]*args[0] + beami["ky"]*args[1] + beami["kz"]*args[2])) * sec_range.astype(complex)
    #APWS_sec = beami["APWS"] * np.exp(1j*(beami["kz"]*args[2])) * sec_range.astype(complex)
    coupling_sec = np.absolute(np.sum(APWS_sec * sec_range.astype(complex)))**2 / (np.sum(np.absolute(beami["APWS"])**2) * np.sum(sec_range**2))

    coupling_loss = 1 - coupling_sec
    return coupling_loss

def nested_secondary_coupling(FPG2, beam, i_kid, make_movie, moviepath, tonelabel):

    global sec_range
    global APWS_sec
    global angle_offset

    global beami
    global fit_param
    beami = beam
    fit_param = FPG2

    args = [fit_param["x_0"],
            fit_param["y_0"],
            fit_param["z_0"],
            fit_param["theta_x0"] + angle_offset,
            fit_param["theta_y0"] + angle_offset]

    optargs = optimize.fmin(func=fit_sec,
                            x0=args,
                            xtol=1e-10,
                            ftol=1e-10,
                            maxiter=10000,
                            maxfun=10000)


    optargs[3] = optargs[3] - angle_offset
    optargs[4] = optargs[4] - angle_offset

    '''
    optargs[2] -= beam["dzxy"]/1000
    print(beam["dzxy"]/1000)
    '''
    out_fits = {}
    out_fits["x_0"] = optargs[0]
    out_fits["y_0"] = optargs[1]
    out_fits["z_0"] = optargs[2]

    out_fits["theta_x0"] = optargs[3]
    out_fits["theta_y0"] = optargs[4]


    # Efficiencies
    APWS_sec_untrunc = beami["APWS"] * np.exp(1j* (beami["kx"]*optargs[0] + beami["ky"]*optargs[1] + beami["kz"]*optargs[2]))

    # Now check for dz at which coupling is 50 % of max value
    # Use steps of 0.001 in z
    coup_max = 1 - fit_sec(optargs)
    coup_d = coup_max

    dz = 0

    # Do one for upper value, one for lower value
    while coup_d > 0.9*coup_max:
        APWS_coup = beami["APWS"] * np.exp(1j* (beami["kx"]*optargs[0] + beami["ky"]*optargs[1] + beami["kz"]*(optargs[2]+dz)))
        coup_d = np.absolute(np.sum(APWS_coup * sec_range.astype(complex)))**2 / (np.sum(np.absolute(beami["APWS"])**2) * np.sum(sec_range**2))

        dz += 0.01

    out_fits["dz_upp"] = dz

    coup_d = coup_max
    dz = 0
    while coup_d > 0.9*coup_max:
        APWS_coup = beami["APWS"] * np.exp(1j* (beami["kx"]*optargs[0] + beami["ky"]*optargs[1] + beami["kz"]*(optargs[2]+dz)))
        coup_d = np.absolute(np.sum(APWS_coup * sec_range.astype(complex)))**2 / (np.sum(np.absolute(beami["APWS"])**2) * np.sum(sec_range**2))

        dz -= 0.01
    out_fits["dz_low"] = dz

    print("Upper dz = {}, lower dz = {}".format(out_fits["dz_upp"], out_fits["dz_low"]))

    # Movie loop
    if make_movie:
        N_frames = int(60 * 15)

        dx = optargs[0] / N_frames
        dy = optargs[1] / N_frames
        dz = optargs[2] / N_frames

        extent = [beami["x"][0,0], beami["x"][0,-1], beami["y"][0,0], beami["y"][-1,0]]

        movie_folder = moviepath + "/{}".format(i_kid)

        if not os.path.isdir(movie_folder):
            os.makedirs(movie_folder)

        for idx in range(N_frames):


            fig, ax = pt.subplots(1,1, figsize=(7,7))

            xn = dx * idx
            yn = dy * idx
            zn = dz * idx

            APWS_plot = beami["APWS"] * np.exp(1j* (beami["kx"]*optargs[0] + beami["ky"]*optargs[1] + beami["kz"]*zn))

            beam_plot = ft.fftshift(ft.fft2(ft.ifftshift(np.squeeze(APWS_plot))))

            to_plot_amp = 20*np.log10(np.absolute(beam_plot) / np.max(np.absolute(beam_plot)))

            ampfig = ax.imshow(to_plot_amp, vmin=-30, vmax=0, origin='lower', interpolation='lanczos', extent=extent)
            #phasefig = ax[1].imshow(np.angle(beam_plot), cmap=cmaps.parula, origin='lower', extent=extent)

            divider1 = make_axes_locatable(ax)

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)

            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')

            ax.set_title(r"$z$ = -{:.2f} mm".format(100*zn), y=1.08)
            ax.set_xlabel(r"$x$ / [mm]")
            ax.set_ylabel(r"$y$ / [mm]")
            ax.set_box_aspect(1)

            pt.savefig(fname= movie_folder + "/{}".format(idx),bbox_inches='tight', dpi=600)
            pt.close()
            pt.cla()
            pt.clf()
            #pt.show()

    APWS_sec_untrunc_pad = beami["APWS_pad"] * np.exp(1j* (beami["kx_pad"]*optargs[0] + beami["ky_pad"]*optargs[1] + beami["kz_pad"]*optargs[2]))

    #maxIdx = np.unravel_index(np.absolute(APWS_sec).argmax(), APWS_sec.shape)
    maxIdx = np.argmax(np.absolute(APWS_sec.flatten('F')))

    beam_focus = ft.fftshift(ft.fft2(ft.ifftshift(np.squeeze(APWS_sec_untrunc_pad))))

    # For phase error calc, remove offset. Not needed for taper/aperture efficiency
    APWS_sec_untrunc = APWS_sec_untrunc * np.exp(-1j*np.angle(APWS_sec_untrunc.flatten('F')[maxIdx]))#[maxIdx[0],maxIdx[1]]))
    APWS_sec = APWS_sec * np.exp(-1j*np.angle(APWS_sec.flatten('F')[maxIdx]))#[maxIdx[0],maxIdx[1]]))

    # On sky beam
    beami["on_sky_beam"] = ft.fftshift(ft.fft2(ft.ifftshift(np.squeeze(APWS_sec))))

    eff_sky = {}
    eff_sky["spillover"] = np.absolute(np.sum(np.conj(APWS_sec) * APWS_sec_untrunc))**2 / (np.sum(np.absolute(APWS_sec_untrunc)**2) * np.sum(np.absolute(APWS_sec)**2))
    eff_sky["coupling_sec"] = np.absolute(np.sum(APWS_sec_untrunc * sec_range))**2 / (np.sum(np.absolute(APWS_sec_untrunc)**2) * np.sum(sec_range**2))
    eff_sky["eta_f"] = np.sum(np.absolute(APWS_sec)**2) / np.sum(np.absolute(APWS_sec_untrunc)**2)
    eff_sky["eta_taper"] = np.absolute(np.sum(APWS_sec_untrunc * sec_range))**2 / np.sum(np.absolute(APWS_sec_untrunc[sec_range])**2) / np.sum(sec_range**2)

    eff_sky["eta_ap"] = eff_sky["eta_taper"] * eff_sky["spillover"]

    # Phase error over pupil
    to_calc = APWS_sec[sec_range]
    eff_sky["rms_phase_error"] = np.std(np.angle(to_calc), ddof=1)

    # Phase error over pupil, weighted by illumination
    wgt = np.absolute(APWS_sec)**2
    wgt = wgt / np.sum(wgt) * np.sum(sec_range)
    to_calc = np.angle(to_calc) * wgt[sec_range]

    eff_sky["rms_wgt_phase_error"] = np.std(to_calc, ddof=1)
    eff_sky["wfe_wgt"] = eff_sky["rms_wgt_phase_error"] * beami["lambda"] / (2*np.pi*1e6) # Total wavefront error
    eff_sky["wfe"] = eff_sky["rms_phase_error"] * beami["lambda"] / (2*np.pi*1e6)

    eff_sky["phase_eff"] = np.exp(-eff_sky["rms_phase_error"]**2)
    eff_sky["phase_wgt_eff"] = np.exp(-eff_sky["rms_wgt_phase_error"]**2)

    # On-sky main beam pattern efficiency
    rnagy = np.sqrt(beami["theta_x_p"]**2 + beami["theta_y_p"]**2) < (fit_param["lambda_D"] * 1.22)
    eff_sky["main_beam_pat_eff"] = np.sum(np.absolute(beami["on_sky_beam"][rnagy])**2, axis=0) / np.sum(np.absolute(beami["on_sky_beam"])**2, axis=0)

    #### Here we initialize uncorrected coupling
    open_ang = np.degrees(np.arctan(200 / 2106)) #
    cond1_u = np.sqrt(beami["theta_x"]**2 + beami["theta_y"]**2) < fit_param["sec_ang"]
    cond2_u = np.sqrt(beami["theta_x"]**2 + beami["theta_y"]**2) > fit_param["blk_ang"]
    sec_range_u = cond1_u & cond2_u

    cond_open = np.sqrt(beami["theta_x"]**2 + beami["theta_y"]**2) < open_ang

    APWS_sec_uncor = beami["APWS"] * np.exp(1j*(beami["kx"]*optargs[0] + beami["ky"]*optargs[1] + beami["kz"]*optargs[2])) * sec_range_u.astype(complex)

    padding_range = (1000, 1000)
    noise_level = 1e-12 + 1j * 1e-12

    M_p = 25.366
    f_pri = 3500 * 1e-3 # m
    f_sec = 88780.076 * 1e-3 #m

    f_sys = M_p * f_pri

    kx_pad = np.pad(beami["kx"], padding_range, 'reflect', reflect_type='odd')
    ky_pad = np.pad(beami["ky"], padding_range, 'reflect', reflect_type='odd')
    kz_pad = np.pad(beami["kz"], padding_range, 'reflect', reflect_type='odd')

    tx_pad = np.degrees(np.arctan(kx_pad / beami["k"]))
    ty_pad = np.degrees(np.arctan(ky_pad / beami["k"]))

    # no chop
    x_center = 118
    y_center = 80

    x_interp = np.linspace(np.min(beami["x"]) - x_center, np.max(beami["x"]) - x_center, kx_pad.shape[0])
    y_interp = np.linspace(np.min(beami["x"]) - x_center, np.max(beami["x"]) - x_center, kx_pad.shape[1])

    x_interp, y_interp = np.meshgrid(x_interp, y_interp)

    tx_pad_p = np.degrees(x_interp*1e-3 / f_sys) * 3600
    ty_pad_p = np.degrees(y_interp*1e-3 / f_sys) * 3600

    cond1_up = np.sqrt(tx_pad**2 + ty_pad**2) < fit_param["sec_ang"]
    cond2_up = np.sqrt(tx_pad**2 + ty_pad**2) > fit_param["blk_ang"]

    sec_range_up = cond1_up & cond2_up

    APWS_test = np.pad(beami["APWS"], padding_range, 'constant', constant_values=(noise_level, noise_level))

    APWS_test = APWS_test * np.exp(1j*(kx_pad*optargs[0] + ky_pad*optargs[1] + kz_pad*optargs[2])) * sec_range_up.astype(complex)

    toWrite_beam = ft.fftshift(ft.ifft2(ft.ifftshift(np.squeeze(APWS_test))))
    toWrite_gridx = tx_pad_p
    toWrite_gridy = ty_pad_p

    folderpath = 'pad_beams/{}/'.format(round(tonelabel))
    exists_folder = os.path.isdir(folderpath)

    if not exists_folder:
        os.makedirs(folderpath)

    np.save(folderpath + "beam", toWrite_beam)
    np.save(folderpath + "grid_x", toWrite_gridx)
    np.save(folderpath + "grid_y", toWrite_gridy)

    maxIdx_u = np.argmax(np.absolute(APWS_sec_uncor.flatten('F')))
    APWS_sec_uncor = APWS_sec_uncor * np.exp(-1j*np.angle(APWS_sec_uncor.flatten('F')[maxIdx_u]))

    beami["on_sky_beam_u"] = ft.fftshift(ft.fft2(ft.ifftshift(np.squeeze(APWS_sec_uncor))))

    eff_sky_u = {}
    eff_sky_u["spillover"] = np.absolute(np.sum(np.conj(APWS_sec_uncor) * APWS_sec_untrunc * cond_open))**2 / (np.sum(np.absolute(APWS_sec_untrunc * cond_open)**2) * np.sum(np.absolute(APWS_sec_uncor)**2))
    eff_sky_u["coupling_sec"] = np.absolute(np.sum(APWS_sec_untrunc * sec_range_u))**2 / (np.sum(np.absolute(APWS_sec_untrunc)**2) * np.sum(sec_range**2))
    eff_sky_u["eta_f"] = np.sum(np.absolute(APWS_sec_uncor)**2) / np.sum(np.absolute(APWS_sec_untrunc)**2)
    eff_sky_u["eta_taper"] = np.absolute(np.sum(APWS_sec_untrunc * sec_range_u))**2 / np.sum(np.absolute(APWS_sec_untrunc[sec_range_u])**2) / np.sum(sec_range_u**2)

    eff_sky_u["eta_ap"] = eff_sky_u["eta_taper"] * eff_sky_u["spillover"]

    rnagy_u = np.sqrt(beami["theta_x_p"]**2 + beami["theta_y_p"]**2) < (fit_param["lambda_D"] * 1.22)
    eff_sky_u["main_beam_pat_eff"] = np.sum(np.absolute(beami["on_sky_beam_u"][rnagy_u])**2) / np.sum(np.absolute(beami["on_sky_beam_u"])**2)
    ####

    print(out_fits["z_0"])
    """
    fig, ax = pt.subplots(1,2)
    ax[0].pcolormesh(beami["theta_x"], beami["theta_y"], np.angle(APWS_sec_untrunc * sec_range_u))
    ax[1].pcolormesh(beami["theta_x"], beami["theta_y"], 20 * np.log10(np.absolute(APWS_sec_untrunc) / np.max(np.absolute(APWS_sec_untrunc))) * sec_range_u, vmin=-30, vmax=0)
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    pt.show()
    """

    return out_fits, eff_sky, eff_sky_u, beami, beam_focus
