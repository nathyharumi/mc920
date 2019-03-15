import cv2
import numpy as np

## I/O images
def get_image(input_text):
    return cv2.imread(input(input_text))

def save_image(image):
    return cv2.imwrite("result_image.png", image)

## Option switcher
def exec_option(option):
    return {
    1 : brightness,
    2 : bit_plane,
    3 : mosaic,
    4 : images_combination}.get(option, invalid_option)()

def brightness():
    image = get_image("Enter the image filename: ")
    gamma = float(input("Enter the gamma value: "))
    brightness_adjustment = ((image/255) ** (1/gamma))*255
    return save_image(brightness_adjustment)

def bit_plane():
    image = get_image("Enter the image filename: ")
    bit_plane_number = float(input("Enter the bit plane number: "))
    result = image & bit_plane_mask(bit_plane_number)
    save_image(result)
    exit()

def mosaic():
    image = get_image("Enter the image filename: ")
    mosaic_block_height = image.shape[0]/4
    mosaic_block_width = image.shape[1]/4
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

## Auxiliar functions

# Bit plane mask switcher
def bit_plane_mask(bit_plane_number):
    return {
    0: 1,  #00000001
    1: 2,  #00000010
    2: 4,  #00000100
    3: 8,  #00001000
    4: 16, #00010000
    5: 32, #00100000
    6: 64, #01000000
    7: 128 #10000000
    }.get(bit_plane_number, 255) #11111111

## Regular flow
option = int(input("Enter the number of an option: \n\
1 - Brightness Adjustment\n\
2 - Bit plane slicing\n\
3 - Mosaic\n\
4 - Images Combination\n"))

exec_option(option)
