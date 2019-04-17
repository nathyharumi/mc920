from skimage import io
import numpy as np
import cv2
import os

def normalize(image):
    #image = image-image.min()
    image = 255*image / image.max()
    return image.astype(np.uint8)

def spatial_filter(filename, option):
    # Available kernels
    h1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    h2 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Invalid option treatment
    if option < 1 or option > 3:
        invalid_option()

    image = io.imread(filename)

    # Applying filters
    if option == 1:
        r = cv2.filter2D(image, -1, h1)
        save_image(r, filename, "h1.png")
    elif option == 2:
        r = cv2.filter2D(image, -1, h2)
        save_image(r, filename, "h2.png")
    elif option == 3:
        r1 = cv2.filter2D(image, -1, h3)
        save_image(r1, filename, "h3.png")
        r2 = cv2.filter2D(image, -1, h4)
        save_image(r2, filename, "h4.png")
        r3 = normalize(np.abs(np.sqrt(np.square(cv2.filter2D(image/255., -1, h3)) + np.square(cv2.filter2D(image/255., -1, h4)))))
        save_image(r3, filename, "h3_h4.png")

def frequency_filter(filename, sigma):
    image = io.imread(filename)

    # Gaussian filter
    g_kernel = cv2.getGaussianKernel(image.shape[0],sigma)
    gf = g_kernel*g_kernel.T
    gf = np.fft.fft2(gf)
    gf = np.fft.fftshift(gf)

    # Image
    f = np.fft.fft2(image)
    f = np.fft.fftshift(f)
    ff = f*gf
    result = np.fft.ifftshift(ff)
    result = np.fft.ifft2(result)
    result = normalize(np.abs(result))#.astype(np.uint8)
    sigma_str = ""
    print (str(sigma).split("."))
    for s in str(sigma).split("."):
        sigma_str.join(s).join("-")
    save_image(result, filename, sigma_str+".png")

def save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_"
    io.imsave(saving_filename + sufix, image, check_contrast=False)

def invalid_option():
    print("Not a valid option.")
    exit()
