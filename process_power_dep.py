
#%%
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# use Aktgg as matplotlib backend
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
# set font parameters
plt.rcParams.update({'font.size': 24, 'font.family': 'Arial'})

from utils.utils_anom_diff import *

# main function to run all plots
def main():

    # so the first part is to convert the real data into averaged differential images, sving those into a analysis*dataset file
    data = load_figure_1()
    # plot
    plot_differential_images(data)

    # get azimuthal profiles
    profiles, time_arr, amplitudes, centers, X = get_azim_profiles(data)

    plot_azim_profiles_fits(profiles, time_arr)

    # EK vs time
    time_arr, Ks, EKs, EKxs, EKys, vars, mean_xs, mean_ys, intensities = kurtosis_var_mean(data)

    plt.figure()
    plt.plot(time_arr, EKs, marker='o', linestyle='', color='black', markersize=16,
            markerfacecolor='none', markeredgewidth=4)
    plt.xlabel("Time (ns)")
    plt.ylabel("EK")
    plt.title("EK vs Time")
    plt.xlim(0,6)
    plt.ylim(-0.8, 3.)
    plot_figure_2()
    time_arrs, Eks_all, vars_all, intensities_all, fluences = load_figure_3()
    plot_EK_power_dependencies(time_arrs, Eks_all, fluences)    
    plot_figure_3b()
    load_figure_4a()
    load_figure_4b()
    load_figure_4c()


if __name__ == "__main__":
    main()
# %%

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from utils.utils_anom_diff import *


def add_value_to_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

def remove_delays(dict, cut_key, long_key):
    '''
    IN: dictionary with the averaged negative contrast images + cut_key and long_key

    The function will remove delays that are not useful for the analysis.
    The first delay is always kept, the rest are removed if they are below cut_key or above long_key.
    
    OUT: dictionary with the delays removed
    '''
    corrected_dict = dict.copy()
    keys = list(corrected_dict.keys())
    for key in keys:
        if key < cut_key:
            corrected_dict.pop(key)

        elif key > long_key:
            corrected_dict.pop(key)

    # add back the first delay

    #key_zero = list(dict.keys())[0]
    #corrected_dict[key_zero] = dict[key_zero]

    # print first and last key
    print('First key: ', list(corrected_dict.keys())[0])
    print('Last key: ', list(corrected_dict.keys())[-1])
    return corrected_dict



def create_dataset(folder):

    # introduce a folder, find all 25*.npy files inside and then average them together (some procesing is necessary)

    '''
    This function has the following steps:

    1. Go through all the 25*.npy files in the folder
    2. For each file:
        a. Load the differential images
        b. Find if the contrast is positive or negative at the ROI, swap to negative if necessary
        c. Store the negative contrast in a dictionary (temp), adding the delay as the key
    
    3. If a delay has many images, average them after calculating the negative contrast
    4. Store into a dictionary with the delay as the key
    
    Q: Why not use full_dict? 
    A: Sadly, I can't find a way to use arraySelection to select files in full_dict.
    Assuming you may have files with different delays, this is the best way I can think of.
    '''

    file_list = glob.glob(os.path.join(folder, '25*.npy'))
    temp = {} # dictionary to store the negative contrast images before averaging

    for file in file_list:
        # load file
        data = np.load(file, allow_pickle=True).item() # load using np, dictionary inside
        
        # dict loaded, so now place things into negative contrast
        ROI_pic = make_ROI(10, 20, size = 120) # ROI for cropping the image, in this case there is no cropping
        ROI = make_ROI(50, 50, size = 40) # ROI for the contrast, select area where negative contrast is expected

        keys = list(data.keys()) # list of delays
        ROI_val = [] # list to store the average contrast around the ROI
        for key in keys:
            image = data[key][ROI_pic[1][0]:ROI_pic[1][1], ROI_pic[0][0]:ROI_pic[0][1]] # crop the image
            ROI_im = image[ROI[1][0]:ROI[1][1], ROI[0][0]:ROI[0][1]] # select the ROI

            if ROI_im.max() > abs(ROI_im.min()): # positive contrast                
                ROI_val.append(1/(ROI_im.sum() + 1) - 1)
                image = 1/(image + 1) - 1
            else: # negative contrast
                ROI_val.append(ROI_im.sum())
                image = image
            add_value_to_key(temp, key, image) # add the negative contrast image to the dictionary

    # temp is ready, now average the images for each delay

    negative_contrast = {} # dictionary to store the averaged negative contrast images
    for key in temp.keys():      
        images = temp[key] # list of images for a given delay
        if len(images) > 1:
            av_im = np.mean(images, axis = 0)
        #
        else: # only one image, to not average a row
            av_im = images[0]
        #
        neg_key = -key # human-readable key
        negative_contrast[neg_key] = av_im # add the averaged image to the dictionary

    corrected_dict = remove_delays(negative_contrast, cut_key = -0.1, long_key = 10.5)

    np.save(os.path.join(folder, 'analyzed_dataset.npy'), corrected_dict) # save the dataset
    return corrected_dict # return the dictionary with the averaged negative contrast images


def create_all_datasets():
    # go through all subfolders in main_folder, create dataset for each and save as analysis*dataset.npy

    main_folder = ['power_dep',
                   'rep_rate_dep',
                   ]

    for folder in main_folder:
        # current path + data + folder
        folder_path = os.getcwd() + '\\'+ 'data'+ '\\' + folder + '\\'
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for subfolder in subfolders:
            create_dataset(subfolder)
            print(f"Dataset created for folder: {subfolder}")

folder = r'P:\Lab\Shared\SharedPythonScripts\Publications\WSe2_anomalous_diffusion\data\power_dep\2.220/'
create_all_datasets()

#%%

import numpy as np
import glob

fol = r'P:\Lab\Shared\SharedPythonScripts\Publications\WSe2_anomalous_diffusion\data\power_dep\0.025/'

file = glob.glob(fol + 'analyzed_dataset.npy')[0]
file_2 = glob.glob(fol + 'analysis*dataset*.npy')[0]

file_data = np.load(file, allow_pickle=True).item()
file_2_data = np.load(file_2, allow_pickle=True).item()

# check if they are the same for each key
for key in file_data.keys():
    if key != -0.2 and key != -3.9:
        im1 = file_data[key]
        im2 = file_2_data[key]
        diff = np.abs(im1 - im2)
        max_diff = diff.max()
        print(f"Key: {key}, Max difference: {max_diff}")

# %%

image = file_data[0.0]

plt.figure()
plt.imshow(image, cmap='gray')

image2 = file_2_data[0.0]
plt.figure()
plt.imshow(image2, cmap='gray')

# %%
