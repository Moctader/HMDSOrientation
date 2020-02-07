import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# Add utility functions here


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
    return image


def print_orthogonal(data, invert=True, res=3.2, title=None, cbar=True, savepath=None):
    """Print three orthogonal planes from given 3D-numpy array.

    Set pixel resolution in µm to set axes correctly.

    Parameters
    ----------
    data : 3D numpy array
        Three-dimensional input data array.
    savepath : str
        Full file name for the saved image. If not given, Image is only shown.
        Example: C:/path/data.png
    invert : bool
        Choose whether to invert y-axis of the data
    res : float
        Imaging resolution. Sets tick frequency for plots.
    title : str
        Title for the image.
    cbar : bool
        Choose whether to use colorbar below the images.
    """
    dims = np.array(np.shape(data)) // 2
    dims2 = np.array(np.shape(data))
    x = np.linspace(0, dims2[0], dims2[0])
    y = np.linspace(0, dims2[1], dims2[1])
    z = np.linspace(0, dims2[2], dims2[2])
    scale = 1 / res
    if dims2[0] < 1500 * scale:
        xticks = np.arange(0, dims2[0], 500 * scale)
    else:
        xticks = np.arange(0, dims2[0], 1500 * scale)
    if dims2[1] < 1500 * scale:
        yticks = np.arange(0, dims2[1], 500 * scale)
    else:
        yticks = np.arange(0, dims2[1], 1500 * scale)
    if dims2[2] < 1500 * scale:
        zticks = np.arange(0, dims2[2], 500 * scale)
    else:
        zticks = np.arange(0, dims2[2], 1500 * scale)

    # Plot figure
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(data[:, :, dims[2]].T, cmap='gray')
    if cbar and not isinstance(data[0, 0, dims[2]], np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(data[:, :, dims[2]]), np.max(data[:, :, dims[2]])],
                             orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    plt.title('Transaxial (xy)')
    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(data[:, dims[1], :].T, cmap='gray')
    if cbar and not isinstance(data[0, dims[1], 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(data[:, dims[1], :]), np.max(data[:, dims[1], :])],
                             orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    plt.title('Coronal (xz)')
    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(data[dims[0], :, :].T, cmap='gray')
    if cbar and not isinstance(data[dims[0], 0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(data[dims[0], :, :]), np.max(data[dims[0], :, :])],
                             orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    plt.title('Sagittal (yz)')

    # Give plot a title
    if title is not None:
        plt.suptitle(title)

    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_yticks(zticks)
    ax3.set_xticks(yticks)
    ax3.set_yticks(zticks)

    # Invert y-axis
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    #plt.tight_layout()

    # Save the image
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    plt.show()
