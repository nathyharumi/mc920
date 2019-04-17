from filters import *

filename = input("Enter the image filename: \n")

option = int(input("Enter the number of an option [1-2]: \n"
                   "1 - Spatial filter\n"
                   "2 - Frequency filter\n"))

if option == 1:
    filter_option = int(input("Enter the number of the filter [1-3]: \n"
                              "1 - h1\n"
                              "2 - h2\n"
                              "3 - h3 and h4\n"))
    spatial_filter(filename, filter_option)
elif option == 2:
    sigma = float(input("Enter the Gaussian standard deviation (if negative, it"
                    " is going to be calculated based on the image shape):\n"))
    frequency_filter(filename, sigma)
else:
    invalid_option()
