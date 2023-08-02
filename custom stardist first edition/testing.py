from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
import os
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()

image_dir = "/scratch/awil743/modeltesting/testing"
file_names = sorted(os.listdir(image_dir))

X = sorted(glob("/scratch/awil743/modeltesting/testing/*.tif"))
X = list(map(imread,X))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently

model = StarDist2D(None, name='customStardist', basedir='models')

def example(model, i, show_dist=True):
    img = normalize(X[i], 1, 99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    # Count the number of unique labels (subtract 1 to exclude the background label 0)
    num_objects = len(np.unique(labels)) - 1

    # Get the file name for the current image
    file_name = file_names[i]

    # Get the actual count from the file name
    actual_count = int(file_name.split('count')[0])

    plt.figure(figsize=(13, 10))
    img_show = img if img.ndim == 2 else img[..., 0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    a = plt.axis()
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.axis(a)
    plt.title(f'Predicted Objects: {num_objects}\nActual Objects: {actual_count}', fontsize=16)
    
    plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    
    # Save the figure to your specified directory
    plt.savefig(f"/scratch/awil743/modeltesting/figure_{i}.png", dpi=400)  # Modify the filename as needed

for i in range(4):
    example(model, i)
