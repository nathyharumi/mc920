import numpy as np
import cv2
import os

def space_filter(filename, option):
    # Available kernels
    h1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    h2 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Invalid option treatment
    if option < 1 or option > 3:
        invalid_option()

    image = cv2.imread(filename)

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
        r3 = (r1**2 + r2**2)**(1/2)
        save_image(r3, filename, "h3_h4.png")

def frequency_filter(filename):
    image = cv2.imread(filename)

def save_image(image, filename, sufix):
    if not os.path.exists("results"):
        os.mkdir("results")
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_"
    cv2.imwrite(saving_filename + sufix, image)

def invalid_option():
    print("Not a valid option.")
    exit()
