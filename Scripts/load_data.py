import os
import cv2
import numpy as np
# import glob
# import h5py
# import h5py as h5
# from PIL import Image
# from scipy.io import loadmat

from scipy import ndimage
import matplotlib.pyplot as plt
# from skimage.io import imsave
import time
from Utilities import load, print_orthogonal
from scipy.io import loadmat
import os.path
# from sklearn import datasets, svm, metrics
# from sklearn.model_selection import train_test_split
#
# import keras
# from keras.applications import DenseNet121
#
# from keras.models import Sequential
# from keras.optimizers import Adam, Nadam
# import tensorflow as tf
#
# from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import misc
import scipy.io as sio
import glob
import imageio


def PLM_plot(images):
    fig = plt.figure(1)
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


def save(HMDS_save_train_rotated, files_hmds, data_xz, n_jobs=12):
    """
    Save a volumetric 3D dataset in given directory.
    Parameters
    ----------
    path : str
        Directory for dataset.
    file_name : str
        Prefix for the image filenames.
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(HMDS_save_train_rotated):
        os.makedirs(HMDS_save_train_rotated, exist_ok=True)
    nfiles = np.shape(data_xz)[2]

    if data_xz[0, 0, 0].dtype is bool:
        data_xz = data_xz * 255

        # Parallel saving (nonparallel if n_jobs = 1)
    Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                            (HMDS_save_train_rotated + '/' + files_hmds + str(k).zfill(8) + '.png',
                             data_xz[:, :, k].astype(np.uint8))
                            for k in tqdm(range(nfiles), 'Saving dataset'))



def read_image(image_png_path, image_png_file):

    """Reads image from given path."""
    # Image
    f = os.path.join(image_png_path, image_png_file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return image


if __name__ == '__main__':

    HMDS_path_train = "/data/Repositories/HMDS_orientation/Data/train/hmds/"
    HMDS_path_test = "/data/Repositories/HMDS_orientation/Data/test/"
    HMDS_save_train_rotated = "/data/Repositories/HMDS_orientation/Data/train_rotated/"
    HMDS_save_test_rotated = "/data/Repositories/HMDS_orientation/Data/test_rotated/hmds"
    PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/train/'
    PLM_path_train_png = '/data/Repositories/HMDS_orientation/Stitched_PLI/train_png/'
    image_png_path = ('/data/Repositories/HMDS_orientation/Data/train_rotated/*/')
    image_png_file = ('/data/Repositories/HMDS_orientation/Data/train_rotated/*/*.png')

    n_jobs = 12
    load_hmds = False
    #load_plm = False

    # TODO: Load the PLM images

    files_plm = os.listdir(PLM_path)
    files_plm.sort()
    print(files_plm)
    #if load_plm:
    for filename in sorted(files_plm):
        if filename.endswith('.mat'):
            print(filename)
            annots = sio.loadmat(PLM_path + filename)
            PLM_plot(annots)
            az90 = annots['az90']
            fig = plt.figure(2)
            plt.imshow(az90)
            # average on axis 0
            a = np.array(az90)
            avg_0 = np.mean(a, axis=0)
            avg_1 = np.mean(a, axis=1)
            # plot (depth axis 1, angle)
            fig = plt.figure(3)
            plt.xlabel('Depth')
            plt.ylabel('angle')
            plt.plot(avg_1)
            plt.show()
            os.chdir(PLM_path_train_png)
            os.listdir(PLM_path_train_png)
            cv2.imwrite(f'{filename}.png', az90)





            # Load HMDS

    files = []
    files_hmds = os.listdir(HMDS_path_train)
    files_hmds.sort()
    # train_hmds, test_hmds = train_test_split(files_hmds, test_size=0.2, shuffle=True)
    if load_hmds:
        for i in range(len(files_hmds)):
            data = load(HMDS_path_train + files_hmds[i], n_jobs=n_jobs)
            print_orthogonal(data, title=files_hmds[i])
            data_xz = np.transpose(data, (2, 0, 1))
            save(HMDS_save_train_rotated + files_hmds[i], files_hmds[i] + '_', data_xz)
            read_image(image_png_path, image_png_file)
            print(data.shape)
        print(files)

    # load HMDS train rotated
    #files_roated = []
    #files_hmds_train_rotated = os.listdir(HMDS_save_train_rotated)

    #for main_png_file in range(len(files_hmds_train_rotated)):
        #rotated_train_data = load(HMDS_save_train_rotated + files_hmds_train_rotated[main_png_file], n_jobs=n_jobs)
        #read_image(image_png_path+[main_png_file], image_png_file)
     #   print(main_png_file)


    # List of all training slices (filenames) (X)

    # Create list of groups according to sample name (group)
    # e.g. 6060-26M = 0, = 1, ...

    # Cross-validation split (group k-fold, 5 splits)
    hmds_train = np.array(files_hmds)
    hmds_train = np.repeat(hmds_train, 2)

    plm_images = np.array(files_plm)
    patient_number = [i.split('-', 1)[0] for i in files_hmds]
    patient_number = np.repeat(patient_number, 2)
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(hmds_train, plm_images, patient_number)

    for train_index, val_index in group_kfold.split(hmds_train, plm_images, patient_number):
        print("TRAIN:", train_index, "TEST:", val_index)
        X_train, X_test = hmds_train[train_index], plm_images[val_index]
        y_train, y_test = hmds_train[train_index], plm_images[val_index]
        print(X_train, X_test, y_train, y_test)
