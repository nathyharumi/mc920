import numpy as np
import cv2
import os

def _get_image(filename):
    return cv2.imread(filename, 0)

def _save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_"
    cv2.imwrite(saving_filename + sufix, image)

def _normalize(image):
    image = image-image.min()
    image = 255*image / image.max()
    return image.astype(np.uint8)

def _fft(image):
    return np.fft.fft2(image)

def _fftshift(image):
    return np.fft.fftshift(image)

def _ifft(image):
    return np.fft.ifft2(image)

def _ifftshift(image):
    return np.fft.ifftshift(image)

def _get_spatial_kernel(option):
    return {
    1: np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1],
                  [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]),
    2: np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                  [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256,
    3: np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    4: np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    }.get(option, np.zeros((3,3)))

def _apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def _combine_images(img1, img2):
    return np.abs(np.sqrt(np.square(img1) + np.square(img2)))

def _get_gaussian_filename_sufix(sigma):
    sigma_str = str(sigma).split(".")
    sufix = "gaussian_" + sigma_str[0]
    if len(sigma_str) > 1:
        sufix = sufix + "-" + sigma_str[1]
    return sufix + ".png"

def spatial_filter(filename, option):
    # Invalid option treatment
    if option < 1 or option > 3:
        invalid_option()

    image = cv2.imread(filename, 0)

    # Applying filters
    if option == 1 or option == 2:
        h = _get_spatial_kernel(option)
        r = _apply_filter(image, h)
        _save_image(r, filename, "h" + str(option) + ".png")
        return r
    else:
        h3 = _get_spatial_kernel(option)
        h4 = _get_spatial_kernel(option + 1)

        # Applying h3
        r1 = _apply_filter(image, h3)
        _save_image(r1, filename, "h3.png")

        # Applying h4
        r2 = _apply_filter(image, h4)
        _save_image(r2, filename, "h4.png")

        # Combining h3 and h4 images
        img1 = _apply_filter(image/255., h3)
        img2 = _apply_filter(image/255., h4)
        r3 = _normalize(_combine_images(img1, img2))
        _save_image(r3, filename, "h3_h4.png")
        return r3

def frequency_filter(filename, sigma):
    image = _get_image(filename)

    # Gaussian filter
    g_kernel = cv2.getGaussianKernel(image.shape[0],sigma)
    gf = g_kernel*g_kernel.T

    # FFT and applying Gaussian Filter
    f = _fftshift(_fft(image))
    f = f*gf
    result = _ifft(_ifftshift(f))
    result = _normalize(np.abs(result))

    # Saving image
    _save_image(result, filename, _get_gaussian_filename_sufix(sigma))

    return result

def invalid_option():
    print("Not a valid option.")
    exit()
