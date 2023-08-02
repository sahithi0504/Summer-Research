from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, calculate_extents, random_label_cmap
from stardist.models import Config2D, StarDist2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

X = sorted(glob("/scratch/awil743/modeltesting/images/*.tif"))
Y = sorted(glob("/scratch/awil743/modeltesting/masks/*.tif"))

# for x, y in zip(X, Y):
#     if Path(x).name != Path(y).name:
#         print(f"Mismatch: {Path(x).name}, {Path(y).name}")

# checks to see if each image has a corresponding mask with the same file name
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = list(map(imread, X))
Y = list(map(imread, Y))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
Y = [fill_label_holes(y) for y in Y]

# Function to pad images
def pad_image(img, target_shape=(256, 256)):
    pads = [(0, max(0, target_shape[i]-img.shape[i])) if i < 2 else (0, 0) for i in range(img.ndim)]
    return np.pad(img, pads, mode='constant')

# Padding images
X = [pad_image(x) for x in X]
Y = [pad_image(y) for y in Y]

rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

# print the length of X and Y to ensure the data is loaded properly
print("Total number of images:", len(X))
print("Total number of masks:", len(Y))

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    n_channel_in = n_channel,
)

model = StarDist2D(conf, name='customStardist', basedir='models')

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

epochs=150
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=epochs)
model.optimize_thresholds(X_val, Y_val)