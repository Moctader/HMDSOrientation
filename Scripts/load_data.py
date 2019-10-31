import os
import cv2
import numpy as np
from Components.Utilities import load, print_orthogonal

if __name__ == '__main__':
    path = "/data/Repositories/HMDS_orientation/Data/5922-4L"
    n_jobs = 27

    data = load(path, n_jobs=n_jobs)

    print_orthogonal(data)
    print(data.shape)

import glob
filenames = glob.glob("/path/to/files/raw/*.tif")
len(filenames)
for fn in filenames:
    img = imageio.imread(fn)
    index = get_location_from_filename(fn)  # We need to write this function
    full_array[index, :, :, :] = img

