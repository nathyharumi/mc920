"""
Trabalho 2 - Pontilhamento de imagens
Nathália Harumi Kuromiya
RA 175188
"""
import cv2
import os
import numpy as np

def _get_image(filename):
    """ Get a .PGM image by its filepath

        Args:
        filename (str): the filepath of the image
    """
    return cv2.imread(filename, -1)

def _save_image(image, filename, sufix):
    """ Save images in .PBM at "results" directory based on its previous filename

        Args:
        image (np.ndarray): image to be saved
        filename (str): the filepath of the image
        sufix (str): the sufix to be add to the new filename

    """

    # Create directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Set filepath
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_" + sufix + ".pbm"

    # Save image
    with open(saving_filename, 'w') as fd:
        fd.write("P4\n{} {}\n".format(image.shape[1], image.shape[0]))
        np.packbits(image, axis=-1).tofile(fd)

def _minmax(pixel):
    """ Nomalize pixel from 0 to 255 by min-max method

        Args:
        pixel (int): the pixel to be normalized

        Returns:
        int: pixel normalized
    """
    if pixel > 255:
        pixel = 255
    if pixel < 0:
        pixel = 0
    return pixel

def _transform(image, max):
    """ Transform pixels from [0, 255] to [0, max]

        Args:
        image (np.ndarray): image to be transformed
        max (int): the max value in the grayscale

        Returns:
        np.ndarray: transformed image
    """
    return np.multiply(image, max/255)

def _invert(image, max):
    """ Invert a binary image

        Args:
        image (np.ndarray): binary image
        max (int): the binary number representing the max number of the grayscale

        Returns:
        np.ndarray: inverted image
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                image[i][j] = max
            elif image[i][j] == max:
                image[i][j] = 0
    return image

def _get_kernel(option):
    """ Get kernel for ordered dithering

        Args:
        option (int): the option number that represents the required kernel

        Returns:
        np.array: kernel required
        None: invalid option

    """
    return {
        1: np.array([[6, 8, 4], [1, 0, 3], [5, 2, 7]]),
        2: np.array([[0, 12, 3, 15], [8, 4, 11, 7], [2, 14, 1, 13], [10, 6, 9, 5]])
        }.get(option, None)

def _get_kernel_sufix(option):
    """ Get filename sufix based on the ordered dithering kernel

        Args:
        option (int) the option number that represents the kernel

        Returns:
        str: the sufix linked to the kernel
        None: invalid option
    """
    return {
        1: "simple_kernel",
        2: "bayer_kernel"
        }.get(option, None)

def ordered_dithering(filename, kernel_option):
    """ Apply ordered dithering

        Args:
        filename (str): the image filepath
        kernel_option (int): the option number of the kernel
    """
    image = _get_image(filename)
    kernel = _get_kernel(kernel_option)
    sufix = _get_kernel_sufix(kernel_option)
    result = np.zeros(image.shape, dtype=np.int16)
    (X, Y) = image.shape
    (KX, KY) = kernel.shape

    if kernel is None:
        exit()

    transformed_image = _transform(image, KX*KY + 1)

    for i in range(X):
        for j in range (Y):
            if transformed_image[i][j] < kernel[i % KX][j % KY]:
                result[i][j] = 0
            else:
                result[i][j] = 1

    result = _invert(result, 1)
    _save_image(result, filename, sufix)

def floyd_steinberg_dithering(filename):
    """ Apply the Floyd-Steinberg dithering method

        Args:
        filename (str): the image filepath
    """
    image = _get_image(filename)
    (X, Y) = image.shape

    for i in range(X-1):
        for j in range(1, Y-1):
            # Binarize image
            old_pixel = image[i][j]
            new_pixel = np.round(old_pixel/255.0) * 255
            image[i][j] = new_pixel

            # Distribute the error
            error = old_pixel - new_pixel
            if j + 1 < Y:
                image[i][j+1] = _minmax(image[i][j+1] + error*7/16.0)
            if i + 1 < X:
                image[i + 1][j] = _minmax(image[i + 1][j] + error*5/16)
                if j > 0:
                    image[i + 1][j - 1] = _minmax(image[i + 1][j - 1] + error*3/16)
                if j + 1 < Y:
                    image[i + 1][j + 1] = _minmax(image[i + 1][j + 1] + error*1/16)

    result = _invert(image, 255)
    _save_image(result, filename, "floyd_steinberg")

def main():
    dithering_method = int(input("Enter a dithering method [1-3]:\n"
                                 "1 - Ordered dithering\n"
                                 "2 - Floyd-Steinberg\n"
                                 "3 - Apply all"))
    filename = input("Enter the filepath of the image:\n")

    if dithering_method == 1:
        kernel_option = int(input("Enter the number of a kernel option [1-2]: \n"
                           "1 - Simple Kernel\n"
                           "[6, 8, 4]\n"
                           "[1, 0, 3]\n"
                           "[5, 2, 7]\n"
                           "2 - Bayer Kernel\n"))
        ordered_dithering(filename, kernel_option)
    elif dithering_method == 2:
        floyd_steinberg_dithering(filename)
    elif dithering_method == 3:
        ordered_dithering(filename, 1)
        ordered_dithering(filename, 2)
        floyd_steinberg_dithering(filename)
    else:
        exit()

if __name__ == '__main__':
    main()
