"""
This code is usefull when result frames saved as images in a folder.
This code combines these images and convert them to a movie. 
"""

import cv2
import numpy as np

movie_path = "/home/alperenlcr/bitirme/full_dependent.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(movie_path, fourcc, 1/0.2, (128, 128))
for i in range(1, 246):

    res_path = f"/home/alperenlcr/bitirme/results_dependent{i}.png"

    images = cv2.imread(res_path) # this is a combined image
    # img1_1, img1_2, img1_3
    # img2_1, img2_2, img2_3
    # ...
    # imgN_1, imgN_2, imgN_3

    # every single image is 128x128
    # this is a combined image
    # I want to take third image of each row and convert it to a movie

    img_list = []
    for i in range(images.shape[0]//128):
        img = images[i*128:(i+1)*128, 256:384]
        img_list.append(img)

    for img in img_list:
        out.write(img)

out.release()
print("Movie is saved to {}".format(movie_path))

