import os
import pickle
import fnmatch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def num2name(n):
    if n == 0:
        return "circle"
    elif n == 1:
        return "cross"
    elif n == 2:
        return "rhombus"
    elif n == 3:
        return "square"
    elif n == 4:
        return "squiggly"
    elif n == 5:
        return "triangle"
    else:
        raise ValueError

def grayscale_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def get_len(dir):
    return len(fnmatch.filter(os.listdir(dir), '*.png'))

def get_img_paths(dir, pattern='*.png'):
    img_paths = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                img_paths.append(os.path.join(path, name))
    return img_paths

def img_to_array(data_path, grayscale=False, desired_size=None, view=False):
    """
    Util function for loading RGB image into 4D numpy array.
    Returns array of shape (1, H, W, C)
    """
    img = Image.open(data_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()

    # preprocess    
    x = np.asarray(img, dtype='float32')
    if grayscale:
        x = np.expand_dims(x, axis=2)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    return x

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs
    ------
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    Returns
    -------
    - grid
    References
    ----------
    - Adapted from CS231n - http://cs231n.github.io/
    """

    (N, H, W, C) = Xs.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H

    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def view_images(X, ubound=1.0, save=False, name=''):
    """ Quick helper function to view rgb or gray images."""
    if X.shape[-1] == 1:
        grid = visualize_grid(X, ubound)
        H, W, C = grid.shape
        grid = grid.reshape((H, W))
        plt.imshow(grid, cmap="Greys_r")
        plt.axis('off')
        if save:
            plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=300)
        plt.show()
    elif X.shape[-1] == 3:
        grid = visualize_grid(X, ubound)
        plt.imshow(grid)
        if save:
            plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=1000)
        plt.show()
    else:
        raise ValueError