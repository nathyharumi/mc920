import cv2
import numpy as np

def _get_image(filename):
    return cv2.imread(filename, -1)

def _save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_"
    cv2.imwrite(saving_filename + sufix, image)

def _normalize(image, min, max):
    return (image - image.min())*(max - min)/(image.max() - image.min()) + min

def ordered_dithering(filename):
    image = _get_image(filename)
    simple_kernel = np.array([[6, 8, 4], [1, 0, 3], [5, 2, 7]])
    bayer_kernel = np.array([[0, 12, 3, 15], [8, 4, 11, 7], [2, 14, 1, 13], [10, 6, 9, 5]])
    if int(input("Enter the number of a kernel option [1-2]: \n"
                       "1 - Simple Kernel\n"
                       "[6, 8, 4]\n"
                       "[1, 0, 3]\n"
                       "[5, 2, 7]\n"
                       "2 - Bayer Kernel\n")) == 1:
        result = dither(image, simple_kernel)
        _save_image(result, filename, "simple_kernel.bpm")
    else:
        result = dither(image, bayer_kernel)
        _save_image(result, filename, "bayer_kernel.bpm")

def dither(image, kernel):
    result = np.zeros(image.shape)
    (x, y) = image.shape
    normalized_image = _normalize(image, 0, kernel.shape[0]*kernel.shape[1] + 1)
    for i in range(x):
        for j in range (y):
            if normalized_image[x][y] < kernel[x % kernel.shape[0]][y % kernel.shape[1]]:
                result[x][y] = 0
            else:
                result[x][y] = 255
    return result
