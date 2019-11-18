import os
import cv2
import numpy as np
import glob
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imsave
import time
from Components.Utilities import load, print_orthogonal


if __name__ == '__main__':

    path = "/data/Repositories/HMDS_orientation/Data/"
    n_jobs = 12

    # Load the HMDS samples
    files = []
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        data = load(path+filelist[i], n_jobs=n_jobs)

        print_orthogonal(data, title=filelist[i], savepath=savepath+i)  # TODO: save the image
        print(data.shape)


    print(files)

    # Alternate way to load
    files = []
    for i in os.listdir(path):
        data = load(path+i, n_jobs=n_jobs)

        print_orthogonal(data)
        print(data.shape)


    print(files)

    # TODO: Load the PLM images

    files = {}

    for filename in os.listdir('/data/Repositories/HMDS_orientation/Data'):
        if os.path.isfile(filename) \
                and f.endswith(".txt") \
                and not filename in files:
            with open(filename, "r") as file:
                files[filename] = file.read()

    for filename, text in files.items():
        print(filename)
        print("=" * 80)
        print(text)



for filename in os.listdir('/data/Repositories/HMDS_orientation/Data'):
    if filename.endswith('.log'):
        with open(os.path.join('/data/Repositories/HMDS_orientation/Data', filename)) as f:
            content = f.read()








#img_dir = "/data/Repositories/HMDS_orientation/Data"
#data_path = os.path.join(img_dir, '*g')
#files = glob.glob(data_path)
#data = []
#for f1 in files:
#    img = cv2.imread(f1)
#   data.append(img)

#print(data)


13.11.2019
import os
import cv2
import numpy as np
import glob
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imsave
import time
from Components.Utilities import load, print_orthogonal


if __name__ == '__main__':

    path = "/data/Repositories/HMDS_orientation/Data/"
    n_jobs = 12

    # Load the HMDS samples
    files = []
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        data = load(path+filelist[i], n_jobs=n_jobs)

        print_orthogonal(data, title=filelist[i], savepath=savepath+i)  # TODO: save the image
        print(data.shape)


    print(files)

    # Alternate way to load
    files = []
    for i in os.listdir(path):
        data = load(path+i, n_jobs=n_jobs)

        print_orthogonal(data)
        print(data.shape)


    print(files)

    # TODO: Load the PLM images

    files = {}

    for filename in os.listdir('/data/Repositories/HMDS_orientation/Data'):
        if os.path.isfile(filename) \
                and f.endswith(".txt") \
                and not filename in files:
            with open(filename, "r") as file:
                files[filename] = file.read()

    for filename, text in files.items():
        print(filename)
        print("=" * 80)
        print(text)



for filename in os.listdir('/data/Repositories/HMDS_orientation/Data'):
    if filename.endswith('.log'):
        with open(os.path.join('/data/Repositories/HMDS_orientation/Data', filename)) as f:
            content = f.read()








#img_dir = "/data/Repositories/HMDS_orientation/Data"
#data_path = os.path.join(img_dir, '*g')
#files = glob.glob(data_path)
#data = []
#for f1 in files:
#    img = cv2.imread(f1)
#   data.append(img)

#print(data)


