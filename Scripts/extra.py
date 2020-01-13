import os
import cv2
import numpy as np
import glob
import h5py
import h5py as h5
from PIL import Image
from scipy.io import loadmat

from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imsave
import time
from Components.Utilities import load, print_orthogonal
from scipy.io import loadmat
import os
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer

def PLM_plot(images):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(images['az90'])
    plt.title('az90')

    plt.subplot(1, 3, 2)
    plt.imshow(images['az'])
    plt.title('az')

    plt.subplot(1, 3, 3)
    plt.imshow(images['ret'])
    plt.title('ret')
    fig.suptitle(filename)
    plt.show()


if __name__ == '__main__':

    HMDS_path = "/data/Repositories/HMDS_orientation/Data/"
    PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/'
    n_jobs = 12

    # TODO: Load the PLM images

    file_list = os.listdir(PLM_path)
    print(file_list)
    for filename in sorted(file_list):
        if filename.endswith('.mat'):
            print(filename)
            annots = loadmat(PLM_path + filename)
            PLM_plot(annots)
            az90 = annots['az90']
            plt.figure()
            plt.imshow(az90)
            # average on axis 0
            a=np.array(az90)
            avg=np.mean(a, axis=0)
            # plot (depth axis 1, angle)
            plt.figure
           # plt.plot(axis=1, avg)




