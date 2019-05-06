import cv2
import os
import numpy as np

def _get_image(filename):
    return cv2.imread(filename, -1)

def _save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_" + sufix
    with open(saving_filename, 'w') as fd:
        fd.write("P4\n{} {}\n".format(image.shape[0], image.shape[1]))
        np.packbits(image, axis=-1).tofile(fd)

def _transform(image, max):
    return np.multiply(image, max/255)

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
        result = _dither(image, simple_kernel)
        _save_image(result, filename, "simple_kernel.pbm")
    else:
        result = _dither(image, bayer_kernel)
        _save_image(result, filename, "bayer_kernel.pbm")

def _dither(image, kernel):
    result = np.zeros(image.shape, dtype=np.int16)
    (x, y) = image.shape
    transformed_image = _transform(image, kernel.shape[0]*kernel.shape[1] + 1)

    for i in range(x):
        for j in range (y):
            if transformed_image[i][j] < kernel[i % kernel.shape[0]][j % kernel.shape[1]]:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result

def fs_dithering(filename):
    image = _get_image(filename)
    result = image//129
    error = image - result


def main():
    dithering_method = int(input("Enter a dithering method [1-2]:\n"
                                 "1 - Ordered dithering\n"
                                 "2 - Floyd-Steinberg\n"))
    filename = input("Enter the filepath of the image:\n")

    if dithering_method == 1:
        ordered_dithering(filename)
    else:
        fs_dithering(filename)

if __name__ == '__main__':
    main()
