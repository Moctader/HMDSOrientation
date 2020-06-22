
import os
import cv2
import numpy as np
from scipy.signal import resample
# import glob
# import h5py
# import h5py as h5
# from PIL import Image
# from scipy.io import loadmat
# from HMDSOrientation.Scripts.load_data import map_uint16_to_uint8, save
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
    end = n * int(len(arr) / n)
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
    for index in range(maximum):  # Go through each index of longest list
        temp = []
        for lst in nested_vals:  # Go through each list
            if index < len(lst):  # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return output


PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/test/'
local_orientation_test = '/data/Repositories/HMDS_orientation/Data/test/local_orientation_test/'
inference_data = '/data/Repositories/HMDS_collagen/workdir/New Folder_old/'  # **/*.png'
HMDS_test = '/data/Repositories/HMDS_orientation/Data/test/hmds_test_flip/'
n_jobs = 12

prediction_false = True


p_prediction=[]
profiles_prediction = []
files_pred = os.listdir(inference_data)
files_pred.sort()
for sample in files_pred:
    prediction = load(inference_data + sample)
    prediction=np.transpose(prediction,(1,0,2))
    prediction = prediction / 65535.
    prediction *= 90
    mean_prediction = np.mean(prediction, axis=0)
    #mean_prediction = np.mean(mean_prediction, axis=1)
    plt.figure()
    plt.imshow(prediction[:,:,0])
    plt.show()
    profiles_prediction.append(mean_prediction)
    p_prediction.append(prediction)

for sample in range(len(profiles_prediction)):
    # Resample
    prediction_resampled = resample(profiles_prediction[sample], len(profiles_prediction[sample]))

# Plot everything
    prediction_linspace_x = np.linspace(0, 2, len(prediction))
    prediction_interp1d_f = interp1d(prediction_linspace_x, prediction_resampled)
    prediction_linspace_xnew = np.linspace(0, 1, num=100, endpoint=True)
    prediction_interp1d_f2 = interp1d(prediction_linspace_x, prediction_resampled)
    plt.plot(prediction_linspace_xnew, prediction_interp1d_f2(prediction_linspace_xnew), label='prediction')
   # plt.legend(loc='best')
    plt.show()

from PIL import Image, ImageEnhance
im = cv2.imread("/data/Repositories/HMDS_collagen/workdir/New Folder_old/5922-4L/5922-4L_00000314.png")

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im[:, :, 1] = im[:, :, 0]
im[:, :, 2] = im[:, :, 0]
im=np.transpose(im, (1,0,2))
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

plt.imshow(adjusted[:, :, 0], cmap='bone')
plt.axis('off')
plt.colorbar()
plt.show()


from PIL import Image, ImageEnhance
im = cv2.imread("/data/Repositories/HMDS_collagen/workdir/New Folder_old/5922-4L/5922-4L_00000100.png")

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
res = cv2.resize(im, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
im[:, :, 1] = im[:, :, 0]
im[:, :, 2] = im[:, :, 0]
im=np.transpose(im, (1,0,2))
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

plt.imshow(adjusted[:, :, 0], cmap='bone',  vmax=250)
plt.axis('off')
plt.colorbar()
plt.show()

im = prediction
im = np.transpose(im, (1, 0, 2))
scale_percent = 0.1
width = int(im.shape[1] * scale_percent)
height = int(im.shape[0] * scale_percent)
dimension = (width, height)
output = cv2.resize(im, dimension, interpolation=cv2.INTER_AREA)
plt.figure()
plt.imshow(output[:, :, 0])
plt.axis('off')
plt.colorbar()
plt.show()


im = cv2.imread("/data/Repositories/HMDS_collagen/workdir/New Folder_old/5922-4L/5922-4L_00000314.png",-1)

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im[:, :, 1] = im[:, :, 0]
im[:, :, 2] = im[:, :, 0]
im=np.transpose(im, (1,0,2))
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
im=im/65535.
im*=90
adjusted = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

plt.imshow(im[:, :, 0], cmap='bone')
plt.axis('off')
plt.colorbar()
plt.show()