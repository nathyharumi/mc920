from morphological_operations import *

# Filepath of images to be tested
filenames = ["../images/pgm/baboon.pgm",
             "../images/pgm/fiducial.pgm",
             "../images/pgm/lena.pgm",
             "../images/pgm/monarch.pgm",
             "../images/pgm/peppers.pgm",
             "../images/pgm/retina.pgm",
             "../images/pgm/sonnet.pgm",
             "../images/pgm/wedge.pgm"]

def main():
    for img in filenames:
        ordered_dithering(img, 1)
        ordered_dithering(img, 2)
        floyd_steinberg_dithering(img)

if __name__ == '__main__':
    main()
