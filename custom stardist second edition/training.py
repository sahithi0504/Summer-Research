from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, calculate_extents, random_label_cmap
from stardist.models import Config2D, StarDist2D
import numpy as np
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching, matching_dataset
import os
import matplotlib.pyplot as plt
import csv
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap
import math
from sklearn.model_selection import train_test_split





########################################################
########################################################

# change seed number below to 42 for the first version
# change seed number below to 101 for the second version
np.random.seed(42)

########################################################
########################################################





lbl_cmap = random_label_cmap()

X_filenames = sorted(glob("/scratch/awil743/modeltesting/images/*.tif"))
Y = sorted(glob("/scratch/awil743/modeltesting/masks/*.tif"))
assert all(Path(x).name==Path(y).name for x,y in zip(X_filenames,Y))

X = list(map(imread, X_filenames))
Y = list(map(imread, Y))

# function for grayscale
def rgb_to_gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# function for grayscale
def ensure_grayscale(X):
    return [rgb_to_gray(x) if x.ndim == 3 else x for x in X]

# ensures that the input images are grayscale
X = ensure_grayscale(X)

# n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
n_channel = 1
axis_norm = (0,1)
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
Y = [fill_label_holes(y) for y in Y]

# function to pad images
def pad_image(img, target_shape=(256, 256)):
    pads = [(0, max(0, target_shape[i]-img.shape[i])) if i < 2 else (0, 0) for i in range(img.ndim)]
    return np.pad(img, pads, mode='constant')

# padding images
X = [pad_image(x) for x in X]
Y = [pad_image(y) for y in Y]

# printing amount of images and masks
print("Total number of images:", len(X))
print("Total number of masks:", len(Y))

n_rays = 32
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    n_channel_in = n_channel,
)

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
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y





################################################################
################################################################

# change number below to 42 for the first version
# change number below to 101 for the second version
rng = np.random.default_rng(42)

# size of dataset
# sizes are 30, 60, 90, 120, 150, 180
# change dataset_size number to the one that was assigned to you
dataset_size = 30

# change version number to 1 for the first version
# change version number to 2 for the second version
version = 1

################################################################
################################################################





# Define a random number generator
rng = np.random.default_rng(version)

# Determine the training and testing sizes
n_train_val = int(0.8 * dataset_size) # 80% of total for training + validation
n_test = dataset_size - n_train_val   # remaining 20% for testing

# Randomly select indices for the training + validation subset from the entire dataset
train_val_indices = rng.choice(len(X), size=n_train_val, replace=False)

# Determine the final training size (85% of train + validation)
n_final_train = int(0.85 * n_train_val)

# Subset for final training and validation
train_indices = train_val_indices[:n_final_train]
val_indices = train_val_indices[n_final_train:]

# Now get your training, validation, and testing datasets
X_train, Y_train = [X[i] for i in train_indices], [Y[i] for i in train_indices]
X_val, Y_val = [X[i] for i in val_indices], [Y[i] for i in val_indices]

# Randomly select indices for the test subset from the remaining data
remaining_indices = list(set(range(len(X))) - set(train_val_indices))
test_indices = rng.choice(remaining_indices, size=n_test, replace=False)

X_test, Y_test = [X[i] for i in test_indices], [Y[i] for i in test_indices]

# prints amount of images in training, validation, and testing
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")






base_dir = "/scratch/awil743/modeltesting/models"
dataset_dir = os.path.join(base_dir, f'datasize_{dataset_size}')
os.makedirs(dataset_dir, exist_ok=True)

# Function to save images in a grid to a PNG file
def save_images_to_file(images, filename, title):
    # Specify the dimensions of the subplot grid
    n = len(images)
    cols = int(math.sqrt(n))  # assuming you want a square grid, change this as per your requirements
    rows = int(math.ceil(n / cols))

    # Create a new figure with specified size
    fig = plt.figure(figsize=(20, 20))  # adjust as needed

    # Set title
    plt.title(title, fontsize=40)  # adjust font size as needed

    # Iterate over each image and add it to the subplot
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i], cmap='gray')  # using gray colormap as these are grayscale images
        ax.axis('off')  # to remove axis

    # Adjust layout and save the figure
    fig.tight_layout()  # adjust layout so labels do not overlap
    fig.savefig(filename, dpi=600)

# Saving the training images
training_filename = os.path.join(dataset_dir, 'training_images.png')  # define the path and name for your image
save_images_to_file(X_train, training_filename, "Training Images")

# Saving the validation images
validation_filename = os.path.join(dataset_dir, 'validation_images.png')  # define the path and name for your image
save_images_to_file(X_val, validation_filename, "Validation Images")

# Saving the testing images
testing_filename = os.path.join(dataset_dir, 'testing_images.png')  # define the path and name for your image
save_images_to_file(X_test, testing_filename, "Testing Images")




# number of epochs
steps = [10, 75, 150, 200, 300]

for i in steps:
    # naming the model
    model_name = 'customStardist_' + str(dataset_size) + '_v' + str(version) + '_epochs_' + str(i)
    model = StarDist2D(conf, name=model_name, basedir=dataset_dir)

    median_size = calculate_extents(list(Y_train), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    epochs = i
    model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=epochs)
    model.optimize_thresholds(X_val, Y_val)



    # Validation Evaluation
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                for x in tqdm(X_val)]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    stats[taus.index(0.5)]

    #plots
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()
    
    # Define the figure filename with the full path
    figure_filename = os.path.join(model.basedir, model_name, "validation_plots.png")

    # Save the figure
    fig.savefig(figure_filename, dpi=300)

    # making a new csv file
    filename = os.path.join(model.basedir, model_name, 'val_stats.csv')

    # Save the data to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['criterion', 'thresh', 'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality', 'by_image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write each DatasetMatching object as a row in the CSV file
        for entry in stats:
            writer.writerow({
                'criterion': entry.criterion,
                'thresh': entry.thresh,
                'fp': entry.fp,
                'tp': entry.tp,
                'fn': entry.fn,
                'precision': entry.precision,
                'recall': entry.recall,
                'accuracy': entry.accuracy,
                'f1': entry.f1,
                'n_true': entry.n_true,
                'n_pred': entry.n_pred,
                'mean_true_score': entry.mean_true_score,
                'mean_matched_score': entry.mean_matched_score,
                'panoptic_quality': entry.panoptic_quality,
                'by_image': entry.by_image,
            })

    

    # Testing Evaluation
    Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                for x in tqdm(X_test)]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats_test = [matching_dataset(Y_test, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    stats_test[taus.index(0.5)]

    #plots
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats_test], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats_test], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()
    
    # Define the figure filename with the full path
    figure_filename = os.path.join(model.basedir, model_name, "test_plots.png")

    # Save the figure
    fig.savefig(figure_filename, dpi=300)


    # making a new csv file
    filename = os.path.join(model.basedir, model_name, 'test_stats.csv')

    # Save the data to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['criterion', 'thresh', 'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality', 'by_image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write each DatasetMatching object as a row in the CSV file
        for entry in stats_test:
            writer.writerow({
                'criterion': entry.criterion,
                'thresh': entry.thresh,
                'fp': entry.fp,
                'tp': entry.tp,
                'fn': entry.fn,
                'precision': entry.precision,
                'recall': entry.recall,
                'accuracy': entry.accuracy,
                'f1': entry.f1,
                'n_true': entry.n_true,
                'n_pred': entry.n_pred,
                'mean_true_score': entry.mean_true_score,
                'mean_matched_score': entry.mean_matched_score,
                'panoptic_quality': entry.panoptic_quality,
                'by_image': entry.by_image,
            })