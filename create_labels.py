import os
import pickle
import numpy as np

from utils import *
from fnmatch import fnmatch

# path params
root_dir = './data/train/'
pattern = '*.png'

# img params
height = 28
width = 28
channels = 1

def get_class(img_path):
    name = img_path.split('/')[-1].split('.')[0].split('_')[0]
    return name.lower()

def label2num(l):
    d = dict([(y, x+1) for x, y in enumerate(sorted(set(l)))])
    return [d[x]-1 for x in l]

def main():

    # crawl subdirectories and grab
    print("[*] Reading...")
    img_paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if fnmatch(name, pattern):
                img_paths.append(os.path.join(path, name))

    num_imgs = len(img_paths)

    # create samples
    X = np.empty((num_imgs, height, width, channels), dtype='float32')
    for i, path in enumerate(img_paths):
        img = img_to_array(path, desired_size=(height, width), grayscale=True)
        X[i] = img

    # create ground truth labels
    print("[*] Creating labels...")
    labels = [get_class(x) for x in img_paths]
    labels = label2num(labels)
    y = np.array(labels)
    y = y.astype('uint8')
    y = np.resize(y, [y.shape[0], 1])

    # pickle dump
    pickle.dump(X, open("./X_train.p", "wb"))
    pickle.dump(y, open("./y_train.p", "wb"))
    print("[*] Done!")

if __name__ == '__main__':
    main()
