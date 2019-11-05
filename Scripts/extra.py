import os
import cv2
import numpy as np
import glob
import h5py
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imsave
import time
from Components.Utilities import load, print_orthogonal


if __name__ == '__main__':

    path = "/data/Repositories/HMDS_orientation/Data/"
    n_jobs = 12


    # Alternate way to load
    files = []
    for i in os.listdir(path):
        data = load(path+i, n_jobs=n_jobs)

        print_orthogonal(data)
        print(data.shape)


    print(files)

    # TODO: Load the PLM images


import scipy.io
mat = scipy.io.loadmat('/data/Repositories/HMDS_orientation/Stitched PLI')












