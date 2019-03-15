import cv2
import numpy as np

def get_image(input_text):
    return cv2.imread(input(input_text))

def save_image(image):
    return cv2.imwrite("result_image.png", image)

def brightness():
    image = get_image("Enter the image filename: ")
    gamma = float(input("Enter the gama value: "))
    brightness_adjustment = ((image/255) ** (1/gamma))*255
    return save_image(brightness_adjustment)

def bit_plane():
    image = get_image("Enter the image filename: ")
    bit_plane_number = float(input("Enter the bit plane number: "))
    exit()

def mosaic():
    image = get_image("Enter the image filename: ")
    exit()
    
def images_combination():
    image_1 = get_image("Enter the first image filename: ")
    image_2 = get_image("Enter the second image filename: ")
    image_1_visibility_rate = float(input("Enter the visibility rate of the \
    first image (0 to 1): "))
    exit()

def invalid_option():
    print("Not a valid option.")
    exit()

def exec_option(option):
    return {
    1 : brightness,
    2 : bit_plane,
    3 : mosaic,
    4 : images_combination}.get(option, invalid_option)()

option = int(input("Enter the number of an option: \n\
1 - Brightness Adjustment\n\
2 - Bit plane slicing\n\
3 - Mosaic\n\
4 - Images Combination\n"))

exec_option(option)
