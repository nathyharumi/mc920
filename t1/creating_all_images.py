from filters import *

baboon_image_path = "../images/baboon.png"
butterfly_image_path = "../images/butterfly.png"
city_image_path = "../images/city.png"
house_image_path = "../images/house.png"
seagull_image_path = "../images/seagull.png"

images_path = [baboon_image_path, butterfly_image_path, city_image_path, house_image_path, seagull_image_path]

for img in images_path:
    for i in range(3):
        spatial_filter(img, i+1)
    frequency_filter(img, 5.4)
