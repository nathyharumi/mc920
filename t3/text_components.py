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

def save_image(image, filename, sufix, extension):
    """ Save images at "results" directory based on its previous filename

        Args:
        image (np.ndarray): image to be saved
        filename (str): the filepath of the image
        sufix (str): the sufix to be add to the new filename

    """

    # Create directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Set filepath
    saving_filename = "results/" + filename.split("/")[-1][:-4] + "_" + sufix + extension

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

def preprocessing(image):
    """ Preprocess the image to make each line as a connected component

    Args:
    image (np.ndarray): original image

    Returns:
    np.ndarray: preprocessed image
    """
    ROW_STRUCTURAL_KERNEL = np.ones((1, 100))
    COLUMN_STRUCTURAL_KERNEL = np.ones((200, 1))
    POS_STRUCTURAL_KERNEL = np.ones((1, 30))

    image = cv2.bitwise_not(image)

    row_closing = closing(image, ROW_STRUCTURAL_KERNEL)
    column_closing = closing(image, COLUMN_STRUCTURAL_KERNEL)

    merged_image = cv2.bitwise_and(row_closing, column_closing)

    return closing(merged_image, POS_STRUCTURAL_KERNEL)

def find_words_preprocessing(image):
    """ Preprocess the image to find words components

    Args:
    image (np.ndarray): original image

    Returns:
    np.ndarray: preprocessed image
    """
    ROW_STRUCTURAL_KERNEL = np.ones((1, 12))
    COLUMN_STRUCTURAL_KERNEL = np.ones((10, 1))

    image = cv2.bitwise_not(image)

    row_closing = closing(image, ROW_STRUCTURAL_KERNEL)
    column_closing = closing(image, COLUMN_STRUCTURAL_KERNEL)

    return cv2.bitwise_or(row_closing, column_closing)

def get_ratios_per_component(image, component):
    """ Get the black pixels ratio (number of black pixels/the area of the
    component) and the transitions ratio (number of transitions black to white
    pixel/number of black pixels)

    Args:
    image (np.ndarray):
    component (list): list containing the x_min, y_min, the width, the high and
    the area of the component

    Returns:
    float: ratio of black pixels
    float: ratio of black-to-white transitions
    """
    area = component[2] * component[3]
    black_pixels = _get_black_pixels_per_component(image, component)
    transitions = _get_transitions_per_component(image, component)

    black_pixels_ratio = black_pixels/area
    transitions_ratio = transitions/black_pixels
    return black_pixels_ratio, transitions_ratio

def _get_black_pixels_per_component(image, component):
    """ Count the number of black pixels in a component area of the image

    Args:
    image (np.ndarray): original image
    component (list): list containing the x_min, y_min, the width, the height
    and the area of the component

    Returns:
    int: number of black pixels
    """
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
    """ Count the number of black-to-white transitions in a component area of
    the image

    Args:
    image (np.ndarray): original image
    component (list): list containing the x_min, y_min, the width, the height
    and the area of the component

    Returns:
    int: number of transitions
    """
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
    """ Get all components coordenates of a image

    Args:
    image (np.ndarray): preprocessed image where to find the components

    Returns:
    list: list of the stats list (x_min, y_min, width, height and area) of all
    components found.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    return stats

def get_text_components(original_image, components):
    """ Filter the text components based on ratio rules.

    Args:
    original_image(np.ndarray): the original image
    components(list): all components

    Returns:
    list: text components

    """
    # Rules
    MIN_BP_RATIO = 0.5 #0.5 pra 0.6 cortou os números
    MAX_BP_RATIO = 0.8 #0.8 pra 0.7 cortou muita linha
    MIN_TR_RATIO = 0.1 #0.1 pra 0.2 cortou muita linha
    MAX_TR_RATIO = 0.6 #0.6 pra 0.5 cortou os números

    text_components = []

    for component in components:
        black_pixels_ratio, transitions_ratio = get_ratios_per_component(original_image, component)
        if MIN_BP_RATIO < black_pixels_ratio and black_pixels_ratio < MAX_BP_RATIO \
        and MIN_TR_RATIO < transitions_ratio and transitions_ratio < MAX_TR_RATIO:
            text_components.append(component)

    return text_components

def highlight_components(original_image, components, pixel_color):
    """ Highlight the components by outlining the components on the original
    image

    Args:
    original_image(np.ndarray): original image
    components(list): components to be outlined
    pixel_color(list or int): color of the outline, depending on the image type

    Returns:
    np.ndarray: image with highlighted components

    """
    image = original_image.copy()
    for component in components:
        x_min, y_min, w, h, area = component
        for i in range(h):
            y = y_min + i
            image[y, x_min] = pixel_color
            image[y, x_min + w - 1] = pixel_color
        for i in range(w):
            x = x_min + i
            image[y_min, x] = pixel_color
            image[y_min + h - 1, x] = pixel_color
    return image

def get_words_components_from_text_components(original_image, text_components):
    """ Get the word components in a original image for each given line
    component

    Args:
    original_image(np.ndarray): original image
    text_components(list): list of line components

    Returns:
    list: word components
    list: number of words per line
    """
    preprocessed_image = find_words_preprocessing(original_image)
    words_components = []
    words_per_line = []
    for component in text_components:
        x_min, y_min, w, h, area = component
        component_slice = preprocessed_image[y_min:y_min + h - 1, x_min:x_min + w - 1].copy()
        w_components = get_components_coordenates(component_slice)[1:]
        for c in w_components:
            c[0] = c[0] + x_min
            c[1] = c[1] + y_min
        words_components.extend(w_components)
        words_per_line.append(len(w_components))
    return words_components, words_per_line

def main():
    filename = "bitmap.pbm"
    original_image = get_image(filename)
    image = preprocessing(original_image)
    components = get_components_coordenates(image)

    # Line components
    text_components = get_text_components(original_image, components)
    highlighted_lines = highlight_components(original_image, text_components, 0)
    save_image(highlighted_lines, filename, "highlighted_lines", ".pbm")

    # Word components
    word_components, words_per_line = get_words_components_from_text_components(
    original_image, text_components)
    highlighted_words = highlight_components(original_image, word_components, 0)
    save_image(highlighted_words, filename, "highlighted_words", ".pbm")

    # Save PNG images for report purposes and better visualization
    image_3_channel = cv2.imread(filename)
    png_lines = highlight_components(image_3_channel, text_components, [255, 0, 0])
    png_words = highlight_components(image_3_channel, word_components, [0, 0, 255])
    save_image(image_3_channel, filename, "original", ".png")
    save_image(png_lines, filename, "highlighted_lines", ".png")
    save_image(png_words, filename, "highlighted_words", ".png")

    print("number of lines: {}, total number of words: {}".format(len(
    text_components), len(word_components)))

if __name__ == '__main__':
    main()
