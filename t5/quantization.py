"""
Trabalho 5 - Quantização
Nathália Harumi Kuromiya
RA 175188
"""

import os
import cv2
import numpy as np
from constants import *

def _save_image(image, filename, sufix):
    """ Save images at "results" directory based on its previous filename

        Args:
        image (np.ndarray): image to be saved
        filename (str): the filepath of the image
        sufix (str): the sufix to be add to the new filename

    """

    # Create directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Set filepath
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_" + sufix +\
    ".png"

    # Save image
    cv2.imwrite(saving_filename, image)

def _preprocess(img):
    """ Preprocess image to apply cv2.kmeans on it.

        Args:
        img (np.ndarray): original image

        Returns
        np.ndarray: preprocessed image
    """

    # Reshape and transform image from np.int8 to np.float32
    return np.float32(img.reshape((-1, 3)))

def _clusterize(img, k):
    """ Cluster images applying cv2.kmeans.

        Args:
        img (np.ndarray): preprocessed image
        k (int): quantity of colors to be used when clustered

        Returns:
        float: ∑ ∥samples − centers(labels)∥ ** 2
        np.array: best labels
        np.array: center of each label
    """
    return cv2.kmeans(img, k, None, (cv2.TERM_CRITERIA_EPS + \
    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

def _reconstruct(shape, label, center):
    """ Reconstruct the image based on the clustering

        Args:
        shape (np.ndarray): original image shape
        label: best labels from clustering
        center: center of each label

        Returns:
        np.ndarray: color quantization result image

    """
    return np.uint8(center)[label.flatten()].reshape((shape))

def quantify(filename, k):
    """ Apply color quantization in a image and save the result

        Args:
        filename(str): the filepath of the image
        k: color quantity

        Returns:
        np.ndarray: color quantization result image
    """
    img = cv2.imread(filename)
    preprocessed_img = _preprocess(img)
    ret, label, center = _clusterize(preprocessed_img, k)
    result = _reconstruct(img.shape, label, center)
    _save_image(result, filename, str(k))
    return result

def run_all(filenames = IMAGES_FILENAMES, ks = N_COLORS):
    """ Apply color quantization in one or more images with one or more
        color quantities.

        Args:
        optional filenames(list): the list of filenames of the images
        to which the color quantization will be applied
        optional ks(list): the list of color quantities to cluster the
        image
    """
    for f in filenames:
        for k in ks:
            quantify(f, k)

def main():
    filename = input("Enter the image filename: \n")
    k = int(input("Enter the color quantity: \n"))
    quantify(filename, k)

if __name__ == '__main__':
    main()
