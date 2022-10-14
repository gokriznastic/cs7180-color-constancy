''' Gopal Krishna
10/13/2022
CS 7180 Advanced Perception '''

import glob
import numpy as np
from scipy.io import loadmat

def get_images_fullpath(img_folder_path):
    """ get all the images for the specified folder
    """

    images_fullpath = glob.glob(img_folder_path + "**/*.png", recursive=True)
    images_fullpath.sort(key=lambda x: x.split('/')[-1].split('.')[0])
    return images_fullpath

def load_groundtruth_illuminant(file_path):
    """ load ground truth illuminant
    """

    real_illum_568 = loadmat(file_path)
    real_rgb = real_illum_568["real_rgb"]
    real_rgb = real_rgb / real_rgb[:, 1][:, np.newaxis]  # convert to chromaticity
    return real_rgb