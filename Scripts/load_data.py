import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from Utilities import load, print_orthogonal, read_image
from scipy.io import loadmat
#from HMDSOrientation.Scripts import map_unit16_to_unit8
import os.path
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




def map_uint16_to_uint8(files_hmds, lower_bound=0, upper_bound=40000):
    """
    Map a 16-bit image trough a lookup table to convert it to 8-bit.
​
    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)
​
    Returns
    -------
    numpy.ndarray[uint8]
    """
    # Check for errors
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError('"lower_bound" must be in the range [0, 65535]')
    elif not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError('"upper_bound" must be in the range [0, 65535]')
    elif lower_bound >= upper_bound:
        raise ValueError('"lower_bound" must be smaller than "upper_bound"')

    # Automatic scaling if not given
    if lower_bound is None:
        lower_bound = np.min(files_hmds)
    if upper_bound is None:
        upper_bound = np.max(files_hmds)

    # Create lookup that maps 16-bit to 8-bit values
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[files_hmds].astype(np.uint8)



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


def save(HMDS_save_test_rotated, files_hmds, data_xz, n_jobs=12):
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
    if not os.path.exists(HMDS_save_test_rotated):
        os.makedirs(HMDS_save_test_rotated, exist_ok=True)
    nfiles = np.shape(data_xz)[2]

    if data_xz[0, 0, 0].dtype is bool:
        data_xz = data_xz * 255

        # Parallel saving (nonparallel if n_jobs = 1)
    Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                            (HMDS_save_test_rotated + '/' + files_hmds + str(k).zfill(8) + '.png',
                             data_xz[:, :, k].astype(np.uint8))
                            for k in tqdm(range(nfiles), 'Saving dataset'))

def read_image(image_png_path, image_png_file):

    """Reads image from given path."""
    # Image
    f = os.path.join(image_png_path, image_png_file)
    #image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)*2
    image = cv2.imread(f, -1)
    return image


if __name__ == '__main__':

    #HMDS_path_train = "/data/Repositories/HMDS_orientation/Data/train/hmds/"
    HMDS_path_test = "/data/Repositories/HMDS_orientation/Data/Collagen orientation/Collagen_test/"
   # HMDS_path_test = "/data/Repositories/HMDS_orientation/Data/test/"
    #HMDS_save_train_rotated = "/data/Repositories/HMDS_orientation/Data/train_rotated/"
    #HMDS_save_test_rotated = "/data/Repositories/HMDS_orientation/Data/test_rotated/hmds/"
    #HMDS_save_test_rotated = "/data/Repositories/HMDS_orientation/Data/collagen_orientation_rotated/Collagen_rotated_train/"
    HMDS_save_test_rotated = "/data/Repositories/HMDS_orientation/Data/collagen_test_transpose/"
    PLM_path = '/data/Repositories/HMDS_orientation/Stitched_PLI/test/'
   # PLM_path_train_png = '/data/Repositories/HMDS_orientation/Stitched_PLI/train_png/'
   # PLM_path_test_png =glob.glob( '/data/Repositories/HMDS_orientation/Stitched_PLI/plm_test_png/*.png')
    image_png_path = ('/data/Repositories/HMDS_orientation/Data/train/hmds/**/')
    image_png_file = ('/data/Repositories/HMDS_orientation/Data/train/hmds/**/*.tif')

   # img= cv2.imread(PLM_path_test_png)

    n_jobs = 12
    load_hmds = False
    load_plm = True
    # TODO: Load the PLM images

    files_plm = os.listdir(PLM_path)
    files_plm.sort()
    print(files_plm)
if load_plm:
    for filename in sorted(files_plm):
        if filename.endswith('.mat'):
            print(filename)
            annots = sio.loadmat(PLM_path + filename)
           # PLM_plot(annots)
            az90 = annots['az90']
            fig = plt.figure(2)
            plt.imshow(az90, cmap='bone')
            plt.axis('off')
            plt.colorbar()
            plt.show()
           #plt.imshow(mask[:, :, 0], cmap='bone')
            # plt.axis('off')
            #lt.colorbar()
            #lt.show()
            # average on axis  0
            a = np.array(az90)
            avg_0 = np.mean(a, axis=0)
            avg_1 = np.mean(a, axis=1)
            # plot (depth axis 1, angle)
            fig = plt.figure(3)
            plt.xlabel('Depth')
            plt.ylabel('angle')
            plt.plot(avg_1)
            #plt.show()
           # os.chdir(PLM_path_test_png)
           # os.listdir(PLM_path_test_png)
            #cv2.imwrite(f'{filename}.png', az90)


    from PIL import Image, ImageEnhance

    im = Image.open("/data/Repositories/HMDS_orientation/Stitched_PLI/plm_test_png/5923-22Ma.mat.png")
    enhancer = ImageEnhance.Contrast(im)
    enhanced_im = enhancer.enhance(3)
    plt.imshow(enhanced_im, cmap='bone')
    plt.axis('off')
    plt.colorbar()
    plt.show()


# Load HMDS
    files = []
    files_hmds = os.listdir(HMDS_path_test)
    files_hmds.sort()

if load_hmds:
    for i in range(len(files_hmds)):
        data = load(HMDS_path_test + files_hmds[i], n_jobs=n_jobs)
        Image8 = map_uint16_to_uint8(data)
        print_orthogonal(data, title=files_hmds[i])
        data_xz = np.transpose(Image8, (2, 0, 1))
        data_xz = np.flip(data_xz, axis=0)
        data_xz = np.transpose(data, (0, 2, 1))
        print_orthogonal(data_xz, title=files_hmds[i])
        save(HMDS_save_test_rotated + files_hmds[i], files_hmds[i] + '_', data_xz)
        print(data.shape)
    print(files)
