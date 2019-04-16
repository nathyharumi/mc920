from filters import *

filename = input("Enter the image filename: \n")

option = int(input("Enter the number of an option [1-2]: \n\
1 - Space filter\n\
2 - Frequency filter\n"))

if option == 1:
    filter_option = int(input("Enter the number of the filter [1-3]: \n\
1 - h1\n\
2 - h2\n\
3 - h3 and h4\n"))
    space_filter(filename, filter_option)
elif option == 2:
    frequency_filter(filename)
else:
    invalid_option()
