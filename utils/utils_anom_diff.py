import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from scipy.optimize import curve_fit

px_size = 0.09 / 1.5  # um

names_code = '*dataset*.npy'  # pattern to find analyzed dataset files

def gaussian(x, A, x0, sigma,):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def load_figure_1():
    # the folder \data\power_dep_0.470 contains the data for power dependency 0.470
    # current dir
    data_folder = os.getcwd() + '\\data\\power_dep\\0.470\\'
    files = glob.glob(os.path.join(data_folder, names_code))
    # this is the data for the  figure 1
    data = np.load(files[0], allow_pickle=True).item()
    return data

# second, plot differential images

def plot_differential_images(data):
    for i, key in enumerate(data.keys()):
        plt.figure()
        plt.imshow(data[key], cmap='gray')
        plt.title(f"Time: {key} ns")
        # remove axis
        plt.axis('off')
        # add scalebar
        plt.gca().add_artist(ScaleBar(px_size, 'um', location='lower right',
                                      frameon=False, color='red', box_color='black',
                                      box_alpha=1, font_properties={'size': '24'}))
        plt.show()  

def find_center(image):
    # find the center of the image by finding the minimum value
    center = np.unravel_index(np.argmin(image, axis=None), image.shape)
    return center

def azimuthal_average(image, center=None):
    if center is None:
        try:
            center = find_center(-image)
        except RuntimeWarning:
            center = np.unravel_index(np.argmin(image, axis=None), image.shape)

    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)  # Convert to integer indices
    
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    radialprofile = radialprofile[:len(image[0])//2]

    # profile noise around the last 20% of the image
    noise_len = int(0.2 * len(radialprofile))
    noise = radialprofile[-noise_len:].mean()

    radialprofile = radialprofile - noise # remove the noise

    return radialprofile, center 

def get_azim_profiles(corrected_dict, center = None):

    # get the azimuthal profiles for each time delay
    profiles = []
    amplitudes = []
    time_arr = []
    centers = []

    key_0 = list(corrected_dict.keys())[0]
    data_0 = -corrected_dict[key_0]
    center = find_center(data_0) # get the center of the image

    for i, key in enumerate(corrected_dict.keys()):
        data = corrected_dict[key]
        # get the azimuthal average
        if i == 0:
            center = find_center(data)
        else:
            center = center 
        radialprofile, center_ = azimuthal_average(data, center = center)

        profiles.append(radialprofile)
        time_arr.append(float(key))
        centers.append(center_)

        # get the amplitude of the profile
        amplitude = radialprofile.min()
        amplitudes.append(amplitude)

    # convert to numpy arrays
    profiles = np.array(profiles)
    time_arr = np.array(time_arr)
    amplitudes = np.array(amplitudes)
    centers = np.array(centers)

    X = np.arange(len(profiles[0]))*px_size # x axis for the profiles

    return profiles, time_arr, amplitudes, centers, X

def plot_azim_profiles_fits(profiles, time_arr):
    # plot azimuthal profiles and fits
    X = np.arange(len(profiles[0])) * px_size  # in um
    sigma_guess = .5  # initial guess for sigma in um
    for i, profile in enumerate(profiles):
        profile_ = - profile

        plt.figure()
        plt.plot(X, profile_, label='Data', marker='o', linestyle='None', color = 'black', markersize=12,
                 markerfacecolor='none', markeredgewidth=4)
        # fit with gaussian
        from scipy.optimize import curve_fit
        p0 = [profile_.max(), 0, sigma_guess]
        try:
            popt, pcov = curve_fit(gaussian, X, profile_, p0=p0)
        except RuntimeError:
            print(f"Fit failed for time {time_arr[i]} ns")
            continue
        sigma_guess = popt[2]
        plt.plot(X, gaussian(X, *popt), label='Gaussian Fit')
        plt.title(f"Time: {time_arr[i]} ns")
        plt.xlabel("Position (um)")
        plt.ylabel("Intensity (a.u.)")
        #plt.legend()
        plt.show()
        

def make_ROI(x, y, size = 20):
    return [[x, x+size], [y, y+size]]

def kurtosis_var_mean(dict_data):
   # open .npy file
    #data_all = np.load(file, allow_pickle=True).item()
    data_all = dict_data
    time_arr = np.array(list(data_all.keys()))


    intensities = []
    vars = []
    mean_xs = []
    mean_ys = []
    Ks = []
    EKs = []
    EKxs = []
    EKys = []

    for key in list(data_all.keys()):
        data = data_all[key]
        #ROI = make_ROI(0, 0, size = 200)
        #data = -data[ROI[1][0]:ROI[1][1], ROI[0][0]:ROI[0][1]]

        
        small_roi = make_ROI(30, 30, size = 50)
        intensity = data.sum()
        intensity = -intensity
        intensities.append(intensity)

        nx = len(data)
        ny = len(data[0])
        mean_x = 0
        mean_y = 0
        mean_x2 = 0
        mean_y2 = 0
        K = 0
        EK = 0
        EKx = 0
        EKy = 0
        var = 0

        for i in range(0, nx):
            for j in range(0, ny):
                mean_x = mean_x - (i - 0.5*nx + 0.5)*data[i][j]
                mean_y = mean_y - (j - 0.5*ny + 0.5)*data[i][j]
                mean_x2 = mean_x2 - ((i - 0.5*nx + 0.5)**2)*data[i][j]
                mean_y2 = mean_y2 - ((j - 0.5*ny + 0.5)**2)*data[i][j]
        mean_x = mean_x/intensity
        mean_xs.append(mean_x)
        mean_y = mean_y/intensity
        mean_ys.append(mean_y)
        mean_x2 = mean_x2/intensity
        mean_y2 = mean_y2/intensity
        varx = mean_x2 - mean_x**2
        vary = mean_y2 - mean_y**2
        var = varx + vary
        vars.append(var)

        for i in range(0, nx):
            for j in range(0, ny):
                x = (i - 0.5*nx + 0.5) - mean_x
                y = (j - 0.5*ny + 0.5) - mean_y
                r2 = x**2 + y**2
                K = K - ((x**2/varx + y**2/vary)**2)*data[i][j]
                EK = EK - (r2**2)*data[i][j]
                EKx = EKx - (x**4)*data[i][j]
                EKy = EKy - (y**4)*data[i][j]

        Ks.append(K/(intensity))
        EKs.append(EK/(intensity*var**2) - 2)
        EKxs.append(EKx/(intensity*varx**2) - 3)
        EKys.append(EKy/(intensity*vary**2) - 3)

    return time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities



def load_figure_3():
    # load the following datasets:
    fols =[ '0.470', '1.150', '4.000']
    fluences = np.array([0.47, 1.15, 4.0]) * 565.9 * 6/100 # in uJ/cm2, conversion factor measured as the losses in the optical path and then convert to fluence the 5Mz pulse

    time_arrs = []
    Eks_all = []
    vars_all = []
    intensities_all = []

    for name in fols:
        data_folder = os.getcwd() + '\\data\\power_dep\\' + str(name) + '\\'
        # load data
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()

        # get the value of the EK, var and intensity as a function of time
        time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)

        time_arrs.append(time_arr)
        Eks_all.append(EKs)
        vars_all.append(np.array(vars)*px_size**2) # in um^2
        intensities_all.append(intensities)    

    return time_arrs, Eks_all, vars_all, intensities_all, fluences


def plot_EK_power_dependencies(time_arrs, Eks_all, fluences):
    fig, ax = plt.subplots(figsize=(10,8))
    cmap = plt.get_cmap("magma").reversed()
    colors = cmap(np.linspace(0, 1, len(fluences)+2))
    for i, (time_arr, Eks) in enumerate(zip(time_arrs, Eks_all)):
        ax.plot(time_arr, Eks, label=f"{fluences[i]:.0f} µJ/cm²",
                color = colors[i+2], marker ='o', linestyle='', markersize=16,
                markerfacecolor='none', markeredgewidth=4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("EK")
    ax.legend(frameon=False,)
    ax.set_ylim(-1., 2.8)
    ax.set_xlim(-0.1, 5.)

    # change zorder so the lowest power is on top
    for i, line in enumerate(ax.lines):
        line.set_zorder( len(ax.lines) - i)

    #axkurt.set_xticks(np.arange(0, 8, 3))

    # change size of ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=27)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax.set_xlabel('Time (ns)', fontsize=27)
    ax.set_ylabel('Excess Kurtosis', fontsize=27)

    # minor ticks
    ax.minorticks_on()
    #  thinner ticks
    ax.tick_params(axis='both', which='major',
                    direction='in', length=5, width=2,)

    ax.tick_params(axis='both', which='minor',
                        direction='in', length=3, width=1)
    
    plt.show()
# make fig 4

# for forth figure, get fig4a (gaussian fits)


def single_linecut(image, center = None):
    # get the center of the image
    if center is None:
        try:
            center = find_center(-image)
        except RuntimeWarning:
            center = np.unravel_index(np.argmin(image, axis=None), image.shape)

    # get the line cut along the x axis
    linecut = image[int(center[0]), :]

    # remove the noise from the linecut
    noise_len = int(0.2 * len(linecut))
    noise = linecut[-noise_len:].mean()
    linecut = linecut - noise # remove the noise

    return linecut, center

def get_single_linecuts(corrected_dict, center = None):

    # get the line cuts for each time delay
    linecuts = []
    amplitudes = []
    time_arr = []
    centers = []

    key_0 = list(corrected_dict.keys())[0]
    data_0 = corrected_dict[key_0]
    #center = find_center(data_0) # get the center of the image

    # get the center as the minimum value of the first image
    center = np.unravel_index(np.argmin(data_0, axis=None), data_0.shape)

    for i, key in enumerate(corrected_dict.keys()):
        data = corrected_dict[key]
        # get the line cut
        linecut, center_ = single_linecut(data, center = center)

        linecuts.append(linecut)
        time_arr.append(float(key))
        centers.append(center_)

        # get the amplitude of the profile
        amplitude = linecut.min()
        amplitudes.append(amplitude)

    # convert to numpy arrays
    linecuts = np.array(linecuts)
    time_arr = np.array(time_arr)
    amplitudes = np.array(amplitudes)
    centers = np.array(centers)

    X = np.arange(len(linecuts[0]))*px_size # x axis for the profiles

    return linecuts, time_arr, amplitudes, centers, X

def fit_linecuts_gauss(linecuts, X):
    popts = []
    perrs = []
    for i in range(len(linecuts)):
        linecuts[i] = -linecuts[i] # flip upside down
        # remove any offset in the linecuts
        noise = linecuts[i][-20:].mean()
        linecuts[i] = linecuts[i] - noise

        sigma_guess = ( X[-1] - X[0] ) /5
        p0 = [linecuts[i].max(), 0, sigma_guess] # initial guess for the amplitude and sigma
    
        popt, pcov = curve_fit(gaussian, X, linecuts[i], p0=p0, bounds=(0, [np.inf, np.inf, np.inf]))
        # error of the sigma
        perr = np.sqrt(np.diag(pcov))
        popts.append(popt)
        perrs.append(perr)
    popts = np.array(popts)
    perrs = np.array(perrs)

    sigmas = popts[:, 2]
    err_sigmas = perrs[:, 2]

    amplitudes_fit = popts[:, 0]
    err_amplitudes_fit = perrs[:, 0]

    return popts, perrs, sigmas, err_sigmas, 

def load_figure_4a():
    fols =[ '0.525', '1.150', '2.220']
    fluences = np.array([0.525, 1.150, 2.220]) * 565.9 * 6/100 # in uJ/cm2, conversion factor measured as the losses in the optical path and then convert to fluence the 5Mz pulse

    time_arrs = []
    Eks_all = []
    vars_all = []
    intensities_all = []

    cmap = plt.get_cmap('magma').reversed()
    colors = [cmap(i) for i in np.linspace(0, 1, len(fluences)+2)]

    plt.figure(figsize=(10,8))
    for i, name in enumerate(fols):
        data_folder = os.getcwd() + '\\data\\power_dep\\' + str(name) + '\\'
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()

        linecuts, time_arr, amplitudes, centers, X = get_single_linecuts(data)
        # fit to gauss
        popts, perrs, sigmas, err_sigmas = fit_linecuts_gauss(linecuts, X)


        plt.errorbar(time_arr, sigmas**2-sigmas[1]**2, yerr=err_sigmas, marker='o', linestyle='',
                    color=colors[i+1], markersize=16,
                    markerfacecolor='none', markeredgewidth=4, 
                    label =f'{fluences[i]:.0f} uJ/cm²'
                    )
        
        time_arrs.append(time_arr)
        Eks_all.append(sigmas)
        vars_all.append(sigmas**2)
    
    plt.xlabel("Time (ns)")
    plt.ylabel(r"Variance ($µm^2$)")
    #plt.title(f"Fluence: {fluences[fols.index(name)]:.2f} µJ/cm²")
    plt.xlim(0,6)
    plt.ylim(-0.1, 0.8)
    plt.legend()

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    plt.xticks([0, 2.5, 5.0])
    plt.yticks([0, 0.4, 0.8])

    plt.tick_params(axis='both', which='major',
                    direction='in', length=5, width=2,)
    plt.tick_params(axis='both', which='minor',
                        direction='in', length=3, width=1)
        
    plt.show()



def load_figure_4b(): 
    fols =[ '0.525', '1.150', '2.220']
    fluences = np.array([0.525, 1.150, 2.220]) * 565.9 * 6/100 # in uJ/cm2, conversion factor measured as the losses in the optical path and then convert to fluence the 5Mz pulse

    time_arrs = []
    Eks_all = []
    vars_all = []
    intensities_all = []

    cmap = plt.get_cmap('magma').reversed()
    colors = [cmap(i) for i in np.linspace(0, 1, len(fluences)+2)]

    plt.figure(figsize=(10,8))
    for i, name in enumerate(fols):
        data_folder = os.getcwd() + '\\data\\power_dep\\' + str(name) + '\\'
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()

        # get kurtosis, vars ..
        time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)

        time_arrs.append(time_arr)
        Eks_all.append(EKs)
        vars_all.append(vars)
        intensities_all.append(intensities)

        vars_gg = (np.array(vars) * px_size**2)
        aux = vars_gg[0]
        if i == 0:
            aux = 0.5
        vars_gg = vars_gg - aux
        plt.plot(time_arr, vars_gg, linestyle='', marker='o', color=colors[i+1],
                label =f'{fluences[i]:.0f} uJ/cm²',
                markersize=12, markerfacecolor='none', markeredgewidth=3
                )
    plt.xlabel("Time (ns)")
    plt.ylabel(r"Variance $(\mu m^2)$")
    plt.xlim(0,6)
    plt.ylim(-0.1, 4.5)    
    plt.legend(fontsize=16)

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    plt.show()


# one more, 4c for the power dependence of D


# for 4c
def load_figure_4c():
    # get the vars and get fits for the selected fleunces

    fols = ['0.029', '0.125', '0.525', '1.150', '2.220']
    fluences = np.array([0.029, 0.125, 0.525, 1.150, 2.220]) * 565.9 * 6/100 # in uJ/cm2, conversion factor measured as the losses in the optical path and then convert to fluence the 5Mz pulse

    time_fits = 2 # ns

    def diff_1d(t, D):
        return 2*D * t
    def diff_2d(t, D):
        return 4*D * t

    plt.figure( figsize=(10,6) )

    for i, name in enumerate(fols):
        data_folder = os.getcwd() + '\\data\\power_dep\\' + str(name) + '\\'
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()

        # get vars
        time_arr_var, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)

        # get fits, extract linecuts and then fit
        linecuts, time_arr, amplitudes, centers, X = get_single_linecuts(data)
        # fit to gauss
        popts, perrs, sigmas, err_sigmas = fit_linecuts_gauss(linecuts, X)

        # calculate D
        time_fit_index = np.argmin(np.abs(time_arr - time_fits))
        msd = sigmas**2 - sigmas[0]**2
        popt, pcov = curve_fit(diff_1d, time_arr[0:time_fit_index], msd[0:time_fit_index])
        D = popt[0]*10
        D_err = np.sqrt(np.diag(pcov))[0] * 10

        plt.errorbar(fluences[i], D, yerr=D_err, marker='o', color='red', markersize=12,
                 markerfacecolor='none', markeredgewidth=3)
        
        # for the variances calculation
        vars_gg = (np.array(vars) * px_size**2)
        aux = vars_gg[0]
        # to put them next to each other, noise is a bit annoying
        if i == 0:
            aux = 3.8
        if i == 1:
            aux = 4.5
        if i == 2:
            aux = 0.5
        vars_gg = vars_gg - aux
        vars_gg = np.maximum(vars_gg, 0)

        # only grab vars_gg that re above 0 and below 5
        time_arr_gg = np.array(time_arr_var)[vars_gg > 0]
        vars_gg = vars_gg[vars_gg > 0]
        time_arr_gg = time_arr_gg[vars_gg < 5]
        vars_gg = vars_gg[vars_gg < 5]

        order = np.argsort(time_arr_gg)  # order the time array
        time_arr_gg = time_arr_gg[order]
        vars_gg = vars_gg[order]
        time_fit_index = np.argmin(np.abs(time_arr_gg - time_fits))

        popt_var, pcov_var = curve_fit(diff_2d, time_arr_gg[0:time_fit_index], vars_gg[0:time_fit_index])
        D_var = popt_var[0]*10
        D_var_err = np.sqrt(np.diag(pcov_var))[0]  * 10


        plt.errorbar(fluences[i], D_var, yerr=D_var_err, marker='s', color='blue', markersize=12,
                 markerfacecolor='none', markeredgewidth=3)

    plt.xlabel(r"Fluence (uJ/cm$^2$)")
    plt.ylabel(r"D (cm$^2$/s)")
    plt.xscale('log')
    plt.xlim(0.9,100)

    plt.show()

def plot_figure_3b():
    # total intensity vs time for different fluences
    # list all the subdirs in data/power_dep
    current_dir = os.getcwd()
    data_base_folder = current_dir + '\\data\\power_dep\\'
    fols = [name for name in os.listdir(data_base_folder) if os.path.isdir(os.path.join(data_base_folder, name))]
    fluences = np.array([float(fol) * 565.9 * 6/100 for fol in fols]) # in uJ/cm2

    plt.figure()
    tot_int_all = []
    for i, fol in enumerate(fols):
        data_folder = os.getcwd() + '\\data\\power_dep\\' + str(fol) + '\\'
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()

        time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)
        tot_int = np.array(intensities).sum(axis=0)
        tot_int_all.append(tot_int)
    tot_int_all = np.array(tot_int_all)
    plt.plot(fluences, tot_int_all, marker='o', linestyle='', color='black', markersize=16,
             markerfacecolor='none', markeredgewidth=4)
    
    plt.show()
        
def plot_figure_2():
    fols = ['5MHz', '2.5MHz', '0.5MHz']
    plt.figure()
    cmap = plt.get_cmap("magma").reversed()
    colors = [cmap(i) for i in np.linspace(0, 1, len(fols)+2)]
    for i, fol in enumerate(fols):
        data_folder = os.getcwd() + '\\data\\rep_rate_dep\\' + str(fol) 
        file = glob.glob(os.path.join(data_folder, names_code))[0]
        data = np.load(file, allow_pickle=True).item()
        time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)

        plt.plot(time_arr, EKs, marker='o', linestyle='', color=colors[i+2], markersize=16,
                 markerfacecolor='none', markeredgewidth=4, label=f'{fol} Rep')
    plt.xlabel("Time (ns)")
    plt.ylabel("EK")
    plt.xlim(0,6)
    plt.ylim(-0.8, 3.)
    plt.show()  

