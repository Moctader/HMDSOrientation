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
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import keras
from keras.applications import DenseNet121

from keras.models import Sequential
from keras.optimizers import Adam, Nadam
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold


def PLM_plot(images):
    fig=plt.figure(1)
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

    HMDS_path = "/data/Repositories/HMDS_orientation/Data/train/"
    PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/train/'
    n_jobs = 12
    load_hmds = False
    load_plm = False

    # TODO: Load the PLM images

    files_plm = os.listdir(PLM_path)
    files_plm.sort()
    print(files_plm)
    if load_plm:
        for filename in sorted(files_plm):
            if filename.endswith('.mat'):
                print(filename)
                annots = loadmat(PLM_path + filename)
                PLM_plot(annots)
                az90 = annots['az90']
                fig=plt.figure(2)
                plt.imshow(az90)
                # average on axis 0
                a=np.array(az90)
                avg_0=np.mean(a, axis=0)
                avg_1=np.mean(a, axis=1)
                # plot (depth axis 1, angle)
                fig=plt.figure(3)
                plt.xlabel('Depth')
                plt.ylabel('angle')
                plt.plot(avg_1)
                plt.show()


    # Load HMDS
    files = []
    files_hmds = os.listdir(HMDS_path)
    files_hmds.sort()
    # train_hmds, test_hmds = train_test_split(files_hmds, test_size=0.2, shuffle=True)

    if load_hmds:
        for i in range(len(files_hmds)):
            data = load(HMDS_path + files_hmds[i], n_jobs=n_jobs)
            print_orthogonal(data, title=files_hmds[i])
            print(data.shape)

    print(files)

    # List of all training slices (filenames) (X)

    # Create list of groups according to sample name (group)
    # e.g. 6060-26M = 0, = 1, ...


    # Cross-validation split (group k-fold, 5 splits)
    hmds_train=np.array(files_hmds)
    hmds_train_copy=hmds_train

    hmds_train_copy=hmds_train.copy()


    plm_images=np.array(files_plm)
    patient_number = [i.split('-', 1)[0] for i in files_hmds]
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(hmds_train, plm_images, patient_number)

    for train_index, val_index in group_kfold.split(hmds_train, plm_images[:21], patient_number[:21]):
        print("TRAIN:", train_index, "TEST:", val_index)
        X_train, X_test = hmds_train[train_index], plm_images[val_index]
        y_train, y_test = hmds_train[train_index], plm_images[val_index]
        print(X_train, X_test, y_train, y_test)