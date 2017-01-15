from PIL import Image
import cv2
import numpy as np

# process = lambda x, shape : cv2.resize(np.asarray(Image.open(x)), tuple(shape[:2]), interpolation=cv2.INTER_AREA)

process = lambda x, shape : cv2.resize(np.asarray(Image.open(x))[100:120], tuple(shape[:2]), interpolation=cv2.INTER_AREA)
