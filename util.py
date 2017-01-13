from PIL import Image
import cv2
import numpy as np

process = lambda x, shape : cv2.resize(np.asarray(Image.open(x)), tuple(shape[:-1]), interpolation=cv2.INTER_AREA)