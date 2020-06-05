import os
import cv2
import numpy as np
from scipy.signal import resample
# import glob
# import h5py
# import h5py as h5
# from PIL import Image
# from scipy.io import loadmat
#from HMDSOrientation.Scripts.load_data import map_uint16_to_uint8, save
from scipy import ndimage
import matplotlib.pyplot as plt
# from skimage.io import imsave
import time
from Utilities import load, print_orthogonal, read_image
from scipy.io import loadmat
from scipy.interpolate import interp1d
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

def load(path, axis=(1, 2, 0), n_jobs=12):

    """
    Loads an image stack as numpy array.

    Parameters
    ----------
    path : str
        Path to image stack.
    axis : tuple
        Order of loaded sample axes.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    Returns
    -------
    Loaded stack as 3D numpy array.
    """
    files = os.listdir(path)
    files.sort()

    files = [fn for fn in files if not os.path.basename(fn).endswith('.db')]

    # Exclude extra files
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue

    files = newlist[:]  # replace list

    # Load data and get bounding box
    data = Parallel(n_jobs=n_jobs)(delayed(read_image)(path, file) for file in tqdm(files, 'Loading'))
    if axis != (0, 1, 2):
        return np.transpose(np.array(data), axis)

    return np.array(data)


def read_image(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread(f, -1)
    return image

def average(arr, n):
    end =  n * int(len(arr)/n)
    return numpy.mean(arr[:end].reshape(-1, n), 1)

def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return output


PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/test/'
local_orientation_test='/data/Repositories/HMDS_orientation/Data/test/local_orientation_test/'
inference_data='/data/Repositories/HMDS_collagen/workdir/New Folder/' #**/*.png'
HMDS_test='/data/Repositories/HMDS_orientation/Data/test/hmds_test_flip/'
n_jobs = 12


prediction_false=True
local_false=True
plm_false=True

if prediction_false:
    # prediction image from hmds test sample
    profiles_prediction = []
    files_pred = os.listdir(inference_data)
    files_pred.sort()
    for sample in files_pred:
        prediction = load(inference_data + sample)
        mean_prediction = np.mean(prediction, axis=0)
        mean_prediction = np.mean(mean_prediction, axis=1)
        #plt.figure()
        #plt.plot(mean_prediction)
        #plt.show()
        profiles_prediction.append(mean_prediction)

    for sample in range(len(profiles_prediction)):
        # Resample
        prediction_resampled = resample(profiles_prediction[sample], len(profiles_prediction[sample]))

    # Plot everything
        prediction_linspace_x = np.linspace(0, 1, len(prediction_resampled))
        prediction_interp1d_f = interp1d(prediction_linspace_x, prediction_resampled)
        prediction_linspace_xnew = np.linspace(0, 1, num=100, endpoint=True)
        prediction_interp1d_f2 = interp1d(prediction_linspace_x, prediction_resampled)
       # plt.plot(prediction_linspace_xnew, prediction_interp1d_f2(prediction_linspace_xnew), label='prediction')
       # plt.legend(loc='best')
       # plt.show()

if local_false:
    # Local orientation
    profiles_local = []
    samples_local = os.listdir(local_orientation_test)
    samples_local.sort()
    for sample in samples_local:
        local = load(local_orientation_test + sample)
        local = np.transpose(local, (1, 0, 2))
        mean_local = np.mean(local, axis=0)
        mean_local = np.mean(mean_local, axis=1)
       # plt.figure()
       # plt.plot(mean_local)
        #plt.show()
        profiles_local.append(mean_local)

    for sample in range(len(profiles_local)):
        # Resample
        local_resampled = resample(profiles_local[sample], len(profiles_local[sample]))
        local_linspace_x = np.linspace(0, 1, len(local_resampled))
        local_interp1d_f = interp1d(local_linspace_x, local_resampled)
        local_linspace_xnew = np.linspace(0, 1, num=100, endpoint=True)
        local_interp1d_f2 = interp1d(local_linspace_x, local_resampled)
       # plt.plot(local_linspace_xnew, local_interp1d_f2(local_linspace_xnew), label='Local orientation')
        #plt.legend(loc='best')
        #plt.show()


if plm_false:
# PLM
    profiles_plm = []
    files_plm = os.listdir(PLM_path)
    files_plm.sort()
    for filename in sorted(files_plm):
        if filename.endswith('.mat'):
            annots = sio.loadmat(PLM_path + filename)
            az90 = annots['az90']
            array_az90 = np.array(az90)
            transpose_az90=np.transpose(array_az90,(1,0))
            mean_transpose_az90 = np.mean(transpose_az90, axis=0)
            #plt.figure()
            #plt.plot(mean_transpose_az90)
            #plt.show()
            profiles_plm.append(mean_transpose_az90)

    # plot the profiles
    for sample in range(len(profiles_plm)):
        # Resample
        plm_resampled = resample(profiles_plm[sample], len(profiles_plm[sample]))
        # Plot everything
        plm_linspace_x = np.linspace(0, 1, len(plm_resampled))
        plm_interp1d_f = interp1d(plm_linspace_x, plm_resampled)
        plm_linspace_xnew = np.linspace(0, 1, num=100, endpoint=True)
        plm_interp1d_f2 = interp1d(plm_linspace_x, plm_resampled)
        #plt.plot(plm_linspace_xnew, plm_interp1d_f2(plm_linspace_xnew), label='plm')
        #plt.legend(loc='best')
        #plt.show()


plt.figure()
plt.plot(local_linspace_xnew, local_interp1d_f2(local_linspace_xnew), label='local orientation')
plt.plot(prediction_linspace_xnew, prediction_interp1d_f2(prediction_linspace_xnew),label='prediction')
plt.plot(plm_linspace_xnew, plm_interp1d_f2(plm_linspace_xnew),label='PLM')
plt.legend(loc='best')
plt.title('Sample 5922-4L')
plt.xticks([0.0, 1.0], ["Surface",  "Deep"])
plt.ylabel('Orientations (degrees)')
plt.show()





#HMDS test samples
"""
files_test= []
files_hmds = os.listdir(HMDS_test)
files_hmds.sort()
for i in range(len(files_hmds)):
    data_hmds = load(HMDS_test + files_hmds[i], n_jobs=n_jobs)
    data_hmds = np.transpose(data_hmds, (1, 0, 2))
   # plt.figure()
    #plt.plot(data_hmds[:, :, 0])
    #plt.show()
    mean_test = np.mean(data_hmds, axis=0)
    mean_test = np.mean(mean_test, axis=1)
    plt.figure()
    plt.plot(mean_test)
    plt.show()

    files_test.append(mean_test)

#files_test=np.mean(files_test,axis=1)
hmds_x = np.linspace(0,1,3114)
hmds_mean = avgNestedLists(files_test)
hmds_mean = np.asarray(hmds_mean, dtype=np.float32)
#plt.figure()
#plt.plot(hmds_mean)


hmds_f=interp1d(hmds_x, hmds_mean)
hmds_xnew = np.linspace(0, 1, num=100, endpoint=True)
hmds_f2 = interp1d(hmds_x, hmds_mean)
plt.figure()
#plt.plot(hmds_x, hmds_mean, hmds_f2(hmds_xnew))
plt.plot(hmds_xnew, hmds_f(hmds_xnew))
plt.legend(['hmds'],loc='best')
plt.show()
"""


"""
p_x = np.linspace(0,1,4000)
prediction=avgNestedLists(profiles_prediction)
prediction = np.asarray(prediction, dtype=np.float32)
#plt.figure()
#plt.plot(prediction)
#plt.show()

p_f=interp1d(p_x,prediction)
p_xnew = np.linspace(0, 1, num=100, endpoint=True)
p_f2 = interp1d(p_x, prediction)
#plt.figure()
#plt.plot(p_x, prediction, p_f2(p_xnew))
plt.plot(p_xnew, p_f2(p_xnew))
plt.legend(loc='lower center')
plt.show()






# polarized light microscopy as a ground truth but it mismatched with sample location
files_plm = os.listdir(PLM_path)
files_plm.sort()
print(files_plm)
plm = []
for filename in sorted(files_plm):
    if filename.endswith('.mat'):
        annots = sio.loadmat(PLM_path + filename)
        az90 = annots['az90']
        a = np.array(az90)
        a=np.transpose(a,(1,0))
        a = np.mean(a, axis=1)
        #plt.figure()
        #plt.plot(a[:,1])
        #plt.show()
        plm.append(a)

x = np.linspace(0,1,2761)
plm = avgNestedLists(plm)
plm = np.asarray(plm, dtype=np.float32)
f=interp1d(x,plm)
xnew = np.linspace(0, 1, num=100, endpoint=True)
f2 = interp1d(x, plm)
#plt.figure()
#plt.plot(x, plm, f2(xnew))
#plt.show()


# Local orientation sample(ground truth)
files = []
files_local_orientation = os.listdir(local_orientation_test)
files_local_orientation.sort()
for i in range(len(files_local_orientation)):
    data = load(local_orientation_test + files_local_orientation[i], n_jobs=n_jobs)
    data = np.transpose(data, (1, 0, 2))
    #plt.figure()
    #plt.imshow(data[:, :, 0])
    #plt.show()
    files_mean = np.mean(data, axis=0)
    files.append(files_mean)

lo_x = np.linspace(0,1,3114)
local_orientation = avgNestedLists(files)
local_orientation = np.asarray(local_orientation, dtype=np.float32)
lo_f=interp1d(lo_x, local_orientation)
lo_xnew = np.linspace(0, 1, num=100, endpoint=True)
lo_f2 = interp1d(lo_x, local_orientation)
#plt.figure()
#plt.plot(lo_x, local_orientation, lo_f2(lo_xnew))
#plt.show()





# Hold on all the plots on the range 0 to 1
plt.figure()
plt.plot(lo_xnew, lo_f(lo_xnew),'o', p_xnew, p_f(p_xnew),'-',hmds_xnew, hmds_f(hmds_xnew),'.' )
#plt.legend(['local orientation'])
plt.legend(['local orientation','prediction','hmds test'],loc='best')
#plt.plot(p_x, prediction, p_f2(p_xnew), label='prediction')
#plt.legend(['prediction'])
#plt.plot(x, plm, f2(xnew), label='PLM')
#plt.legend(['PLM'])
#plt.plot(hmds_x, hmds_mean, hmds_f2(hmds_xnew))
#plt.legend(['hmds test'])
plt.xticks([0.0, 1.0], ["Surface",  "Deep"])
plt.ylabel('Orientations (degrees)')

plt.show()

"""




