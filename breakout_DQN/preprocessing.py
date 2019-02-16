import numpy as np

from skimage.transform import resize
from skimage.color import rgb2gray

def pre_proc(x):
    width = 84
    height = 84
    x = rgb2gray(x)
