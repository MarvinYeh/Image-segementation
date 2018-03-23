import numpy as np
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def fix_pad_image(img):
    '''pad all input images to (32*k)*(32*k)'''
    h = img.shape[-3]
    w = img.shape[-2]
    pad_w = np.ceil(w/32)*32-w
    pad_h = np.ceil(h/32)*32-h
    npad = ((0,0),(0,pad_w),(0,pad_h),(0,0))
    img_padded = np.pad(img, pad_width=npad,mode='constant',constant_values=0)
    return img
