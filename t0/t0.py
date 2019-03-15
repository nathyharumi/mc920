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
    return save_image(result)

def mosaic():
    image = get_image("Enter the image filename: ")
    sequences = ((1, 6, 16, 5, 8, 9, 12, 7), (2, 11), (3, 13, 4), (10, 14, 15))
    for sequence in sequences:
        switch_blocks_by_seq(image, sequence)
    return save_image(image)

def images_combination():
    image_1 = get_image("Enter the first image filename: ")
    image_2 = get_image("Enter the second image filename: ")
    image_1_visibility_rate = float(input("Enter the visibility rate of the \
    first image (0 to 1): "))
    result = image_1*image_1_visibility_rate + image_2*(1 - image_1_visibility_rate)
    return save_image(result)

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

def get_block_limits(image, block_number):
    block_height = int(image.shape[0]/4)
    block_width = int(image.shape[1]/4)
    block_line = int((block_number-1)/4)
    block_column = (block_number-1)%4
    return {
    "Xi" : block_line*block_height,
    "Xf": (block_line+1)*block_height,
    "Yi": block_column*block_width,
    "Yf": (block_column+1)*block_width
    }

def get_block(image, block_number):
    block_limits = get_block_limits(image, block_number)
    block = image[block_limits.get("Xi", 0):block_limits.get("Xf", 0),\
    block_limits.get("Yi", 0):block_limits.get("Yf", 0)]
    return block

def switch_blocks(image, dest_block_number, source_block):
    dest_block_limits = get_block_limits(image, dest_block_number)
    image[dest_block_limits.get("Xi", 0):dest_block_limits.get("Xf", 0),\
    dest_block_limits.get("Yi", 0):dest_block_limits.get("Yf", 0)] \
    = source_block
    return image

def switch_blocks_by_seq(image, sequence):
    mosaic_block_aux = get_block(image, sequence[0]).copy()
    i = 0
    while i < (len(sequence) - 1):
        image = switch_blocks(image, sequence[i], get_block(image,\
        sequence[i+1]))
        i = i + 1
    image = switch_blocks(image, sequence[len(sequence)-1], mosaic_block_aux)
    return image

## Regular flow
option = int(input("Enter the number of an option: \n\
1 - Brightness Adjustment\n\
2 - Bit plane slicing\n\
3 - Mosaic\n\
4 - Images Combination\n"))

exec_option(option)
