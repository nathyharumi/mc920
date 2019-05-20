"""
Trabalho 3 - Operações Morfológicas
Nathália Harumi Kuromiya
RA 175188
"""
import cv2
import os
import numpy as np

def get_image(filename):
    """ Get a .PBM image by its filepath

        Args:
        filename (str): the filepath of the image

        Returns:
        np.ndarray: image
    """
    return cv2.imread(filename, -1)

def save_image(image, filename, sufix):
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

def closing(image, kernel):
    """ Implement the closing operation using dilate and erode

    Args:
    image (np.ndarray): original image
    kernel (np.array): the structural element kernel

    Returns:
    np.ndarray: image after closing operation
    """
    return cv2.erode(cv2.dilate(image, kernel), kernel)

def row_column_closing(image, row_kernel, column_kernel):
    image = cv2.bitwise_not(image)

    row_closing = closing(image, row_kernel)
    column_closing = closing(image, column_kernel)
    return row_closing, column_closing

def preprocessing(image):
    ROW_STRUCTURAL_KERNEL = np.ones((1, 100))
    COLUMN_STRUCTURAL_KERNEL = np.ones((200, 1))
    POS_STRUCTURAL_KERNEL = np.ones((1, 30))

    row_closing, column_closing = row_column_closing(image,
        ROW_STRUCTURAL_KERNEL, COLUMN_STRUCTURAL_KERNEL)
    merged_image = cv2.bitwise_and(row_closing, column_closing)

    return closing(merged_image, POS_STRUCTURAL_KERNEL)

def find_words_preprocessing(image):
    ROW_STRUCTURAL_KERNEL = np.ones((1, 10))
    COLUMN_STRUCTURAL_KERNEL = np.ones((10, 1))

    row_closing, column_closing = row_column_closing(image,
        ROW_STRUCTURAL_KERNEL, COLUMN_STRUCTURAL_KERNEL)

    return cv2.bitwise_or(row_closing, column_closing)

def get_ratios_per_component(image, component):
    area = component[2] * component[3]
    black_pixels = _get_black_pixels_per_component(image, component)
    transitions = _get_transitions_per_component(image, component)

    black_pixels_ratio = black_pixels/area
    transitions_ratio = transitions/black_pixels
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

def get_components_coordenates(image):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    return stats

def get_text_components(original_image, components):
    # Rules
    MIN_BP_RATIO = 0.2
    MAX_BP_RATIO = 0.9
    MIN_TR_RATIO = 0.0
    MAX_TR_RATIO = 1.0

    text_components = []

    for component in components:
        black_pixels_ratio, transitions_ratio = get_ratios_per_component(original_image, component)
        if MIN_BP_RATIO < black_pixels_ratio and black_pixels_ratio < MAX_BP_RATIO \
        and MIN_TR_RATIO < transitions_ratio and transitions_ratio < MAX_TR_RATIO:
            text_components.append(component)

    return text_components

def highlight_components(PBM_image, components):
    image = PBM_image.copy()
    for component in components:
        x_min, y_min, w, h, area = component
        for i in range(h):
            y = y_min + i
            image[y][x_min] = 0
            image[y][x_min + w - 1] = 0
        for i in range(w):
            x = x_min + i
            image[y_min][x] = 0
            image[y_min + h - 1][x] = 0
    return image

def get_words_components_from_text_components(original_image, text_components):
    preprocessed_image = find_words_preprocessing(original_image)
    words_components = []
    for component in text_components:
        x_min, y_min, w, h, area = component
        words_components.extend(get_components_coordenates(original_image
            [y_min:y_min + h - 1][x_min:x_min + w - 1])[1:])
    return words_components

def main():
    filename = "bitmap.pbm"
    original_image = get_image(filename)
    image = preprocessing(original_image)
    components = get_components_coordenates(image)
    text_components = get_text_components(original_image, components)
    words_components = get_words_components_from_text_components(original_image, text_components)
    highlighted_lines = highlight_components(original_image, text_components)
    highlighted_words = highlight_components(highlighted_lines, words_components)
    save_image(highlighted_lines, filename, "highlighted_lines")
    save_image(highlighted_words, filename, "highlighted_words")


if __name__ == '__main__':
    main()
