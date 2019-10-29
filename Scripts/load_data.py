import os
import cv2
import numpy as np
from Components.Utilities import load, print_orthogonal

if __name__ == '__main__':
    path = "/data/Repositories/HMDS_orientation/Data/5922-4L"
    n_jobs = 12

    data = load(path, n_jobs=n_jobs)

    print_orthogonal(data)

