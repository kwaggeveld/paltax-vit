"""
    A program to process the COSMOS real galaxy dataset and create train- 
    and test datasets to be used as light sources in Paltax.

    Usage:
        python cosmos_catalogue_cuts.py \
            --cosmos_folder=/path/to/cosmos_dataset \
            --bad_galaxies=/path/to/bad_galaxies.csv \
            --val_galaxies=/path/to/validation_galaxies.csv

    Outputs:
        COSMOS_train.h5:    training dataset
        COSMOS_test.h5:     validation/test dataset

    Notes:
        - Requires the Paltas package, which can be downloaded at
          https://github.com/swagnercarena/paltas/
        - The COSMOS real galaxy dataset can be downloaded at https://zenodo.org/records/3242143. 
          Paltax assumes COSMOS_23.5_training_sample.tar.gz.
        - The list of validation and bad galaxies excluded for Paltax can be found in
          https://github.com/swagnercarena/paltas/tree/main/paltas/Sources
        
    Written by Koen Waggeveld, k.c.waggeveld@student.rug.nl.
"""


from paltas.Sources.cosmos import COSMOSExcludeCatalog, COSMOSIncludeCatalog
from scipy.stats import norm
import pandas as pd
import numpy as np
import h5py
from absl import app, flags
from tqdm import trange

# ============================================================
#  Global variables
# ============================================================

CATALOG_PATH = None
NPY_FILES = None

# ============================================================
#  Input parsing
# ============================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('cosmos_folder', None, 'Folder containing the COSMOS real galaxy dataset.')
flags.DEFINE_string('bad_galaxies', None, 'CSV containing galaxies to be included.')
flags.DEFINE_string('val_galaxies', None, 'CSV containing galaxies to be used for validation.')

# ============================================================
#  Functions
# ============================================================

def resize_images(mask, imgsize)
    images = []
    for i in trange(len(mask)):
        if not mask[i]:
            continue

        img = np.load(NPY_FILES + f"/img_{i}.npy")
        canvas = np.zeros((imgsize, imgsize))

        if img.shape[0] < imgsize and img.shape[1] < imgsize:               # Zero-pad the image
            dim1 = imgsize - img.shape[0]  # missing pixels in first dim
            dim2 = imgsize - img.shape[1]  # missing pixels in second dim
            canvas[dim1 // 2 : imgsize - dim1 // 2 - dim1 % 2, dim2 // 2 : imgsize - dim2 // 2 - dim2 % 2] = img
        elif img.shape[0] > imgsize and img.shape[1] > imgsize:             # Crop the image
            dim1 = img.shape[0] - imgsize  # excess pixels in first dim
            dim2 = img.shape[1] - imgsize  # excess pixels in second dim
            canvas = img[dim1 // 2 : dim1 // 2 + imgsize, dim2 // 2 : dim2 // 2 + imgsize]
        
        images.append(canvas)

    return np.array(images)

def build_catalog(catalog, name, imgsize = 256)
    catalog = np.load(CATALOG_PATH, allow_pickle=True)
    mask = catalog._passes_cuts()

    images = resize_images(mask, imgsize)
    pixel_sizes_train = catalog['pixel_width'][mask]
    redshifts_train = catalog['z'][mask]

    # Create an HDF5 file
    with h5py.File(f'{name}.h5', 'w') as h5file:
        h5file.create_dataset('images',      data = images)
        h5file.create_dataset('pixel_sizes', data = pixel_sizes_train)
        h5file.create_dataset('redshifts',   data = redshifts_train)

# ============================================================
#  Main body
# ============================================================

def main(_):
    global CATALOG_PATH, NPY_FILES

    cosmos_folder = FLAGS.cosmos_folder
    bad_galaxies = FLAGS.bad_galaxies
    val_galaxies = FLAGS.val_galaxies
    
    CATALOG_PATH = cosmos_folder + "/paltas_catalog.npy"
    NPY_FILES = cosmos_folder + "/npy_files"

    cosmology_parameters = {
        'cosmology_name': 'planck18'
    }

    source_parameters = {
        'minimum_size_in_pixels': 64,
        'faintest_apparent_mag':  20,
        'max_z':                  1.0,
        'smoothing_sigma':        0.00,
        'cosmos_folder':          cosmos_folder,
        'random_rotation':        True,
        'min_flux_radius':        10.0,
        'center_x':               norm(loc=0.0,scale=0.16).rvs,
        'center_y':               norm(loc=0.0,scale=0.16).rvs,
        'output_ab_zeropoint':    25.127,
        'z_source':               1.5,
    }

    source_parameters_train = source_parameters
    source_parameters['source_exclusion_list'] = np.append(
                            pd.read_csv(bad_galaxies, names = ['catalog_i'])['catalog_i'].to_numpy(),
                            pd.read_csv(val_galaxies, names = ['catalog_i'])['catalog_i'].to_numpy()
                        )

    source_parameters_test = source_parameters
    source_parameters_test['source_inclusion_list'] = pd.read_csv(val_galaxies, names = ['catalog_i'])['catalog_i'].to_numpy()

    imgcatalog_train = COSMOSExcludeCatalog(cosmology_parameters, source_parameters_train)
    imgcatalog_test  = COSMOSIncludeCatalog(cosmology_parameters, source_parameters_test)

    build_catalog(imgcatalog_train, "COSMOS_train")
    build_catalog(imgcatalog_test, "COSMOS_test")

if __name__ == '__main__':
    app.run(main)