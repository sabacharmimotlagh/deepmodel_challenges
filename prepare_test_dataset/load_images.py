import os
import cv2

def load_images(directory,filetype):

    # Loads images from a given directory
    files = os.listdir(directory)

    im = []
    for image in files:
        if filetype in image:
            im.append(cv2.imread(directory + image))
    return im