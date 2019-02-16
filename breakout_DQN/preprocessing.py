import numpy as np

from skimage.transform import resize
from skimage.color import rgb2gray

def preproc(x):
    width = 84
    height = 84
    x_gray = rgb2gray(x)
    x_resize = resize(x_gray, (width, height), mode='constant')
    x_uint8 = np.uint8(x_resize * 255)
    return x_uint8