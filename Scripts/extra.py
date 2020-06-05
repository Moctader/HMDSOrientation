

def save(HMDS_save, files_hmds, data_xz, n_jobs=12):
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
    if not os.path.exists(HMDS_save):
        os.makedirs(HMDS_save, exist_ok=True)
    nfiles = np.shape(data_xz)[2]

    if data_xz[0, 0, 0].dtype is bool:
        data_xz = data_xz * 255

        # Parallel saving (nonparallel if n_jobs = 1)
    Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                            (HMDS_save + '/' + files_hmds + str(k).zfill(8) + '.png', data_xz[:, :, k].astype(np.uint8))
                            for k in tqdm(range(nfiles), 'Saving dataset'))



def save(HMDS_save, files_hmds, data_xz, n_jobs=12):
    """
    Save a volumetric 3D dataset in given directory.
    Parameters
    ----------
    path : str
        Directory for dataset.
    files_hmds:
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(HMDS_save):
        os.makedirs(HMDS_save, exist_ok=True)
    n_slices = np.shape(data_xz)[2]

    if data_xz.dtype is np.bool:
        data_xz = data_xz.astype(np.uint8) * 255

        # Parallel saving (nonparallel if n_jobs = 1)
    Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                            (os.path.join(HMDS_save, files_hmds, str(k).zfill(8) + '.png'),
                             data_xz[:, :, k])
                            for k in tqdm(range(n_slices), 'Saving dataset'))

    # for idx in (files_plm):
    #   os.chdir(PLM_path_train_png)
    #  os.listdir(PLM_path_train_png)
    # cv2.imwrite(f'{idx}.png', az90)
    #  os.listdir(PLM_path_train_png)
