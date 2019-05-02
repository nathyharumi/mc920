import cv2
import numpy as np

def _get_image(filename):
    return cv2.imread(filename, 0)

def _save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_"
    cv2.imwrite(saving_filename + sufix, image)

def _normalize(image, min, max):

def ordered_dithering(image):
    dithering_kernel = np.array([[6, 8, 4], [1, 0, 3], [5, 2, 7]])
    bayer_kernel = np.array([[0, 12, 3, 15], [8, 4, 11, 7], [2, 14, 1, 13], [10, 6, 9, 5]])

    (x, y) = image.shape
    for i in range(x):
        for j in range (y):
            image[i][j] =

def fs_dithering():
