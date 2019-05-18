"""
Trabalho 3 - Operações Morfológicas
Nathália Harumi Kuromiya
RA 175188
"""
import cv2
import os
import numpy as np

def _get_image(filename):
    """ Get a .PBM image by its filepath

        Args:
        filename (str): the filepath of the image

        Returns:
        np.ndarray: image
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
    cv2.imwrite(saving_filename, image)

def _closing(image, kernel):
    """ Implement the closing operation using dilate and erode

    Args:
    image (np.ndarray): original image
    kernel (np.array): the structural element kernel

    Returns:
    np.ndarray: image after closing operation
    """
    return cv2.erode(cv2.dilate(image, kernel), kernel)

def _preprocessing(image):
    ROW_STRUCTURAL_KERNEL = np.ones((1, 100))
    COLUMN_STRUCTURAL_KERNEL = np.ones((200, 1))
    POS_STRUCTURAL_KERNEL = np.ones((1, 30))

    image = cv2.bitwise_not(image)

    row_closing = _closing(image, ROW_STRUCTURAL_KERNEL)
    column_closing = _closing(image, COLUMN_STRUCTURAL_KERNEL)

    merged_image = cv2.bitwise_and(row_closing, column_closing)
    return _closing(merged_image, POS_STRUCTURAL_KERNEL)

def _get_ratios(image, components):
    black_pixels_ratio = []
    transitions_ratio = []
    for component in components:
        area = component[2] * component[3]
        black_pixels = _get_black_pixels_per_component(image, component)
        transitions = _get_transitions_per_component(image, component)

        black_pixels_ratio.append(black_pixels/area)
        transitions_ratio.append(transitions/black_pixels)
    return black_pixels_ratio, transitions_ratio

def _get_black_pixels_per_component(image, component):
    x_min, y_min, w, h, area = component
    black_pixels = 0
    for i in range(h):
        y = y_min + i
        for j in range(w):
            x = x_min + j
            if image[y, x] == 255:
                black_pixels = black_pixels + 1
    return black_pixels

def _get_transitions_per_component(image, component):
    x_min, y_min, w, h, area = component
    transitions = 0
    for i in range(h - 1):
        y = y_min + i
        for j in range(w - 1):
            x = x_min + j
            if bool(image[y, x]) ^ bool(image[y + 1, x]):
                transitions = transitions + 1
            if bool(image[y, x]) ^ bool(image[y, x + 1]):
                transitions = transitions + 1
    return transitions

def _get_components_coordenates(image):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    return stats

def _get_text_components(components, black_pixels_ratios, transitions_ratios):
    return components

def main():
    original_image = _get_image("bitmap.pbm")
    image = _preprocessing(original_image)
    components = _get_components_coordenates(image)
    black_pixels_ratios, transitions_ratios = _get_ratios(original_image, components)
    text_components = _get_text_components(components, black_pixels_ratios, transitions_ratios)

if __name__ == '__main__':
    main()
