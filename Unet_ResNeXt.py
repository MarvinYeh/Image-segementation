from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
from img_utils import *


def NeXt_block(input, iteration,in_channel, cardinality = 32, init_downsample=True):
    x = input
    group_filter_size = int(in_channel // cardinality)
    for i in range(iteration):
        shortcut=x
        agg_conv=[]
        if init_downsample==True:
            shortcut = tf.layers.conv2d(x,in_channel,1,(2,2),padding='same')
            x = tf.layers.conv2d(x,in_channel,1,(2,2),padding='same')
        else:
            x = tf.layers.conv2d(x, in_channel, 1, (1, 1), padding='same')

        for _ in range(cardinality):
            x = tf.layers.conv2d(x,group_filter_size,3,(1,1),padding='same')
            agg_conv.append(x)
        x = tf.concat(agg_conv,-1)
        x = tf.layers.conv2d(x, in_channel*2,1,padding='same')
        x = tf.layers.batch_normalization(x,axis=-1)
        x = tf.nn.relu(x)
        x += shortcut
    return x

class ResneXt_Unet(object):
    '''using ResneXt, aggregational convolution. [Net in Net]
    https://arxiv.org/pdf/1611.05431.pdf
    use Next_block function to build the encoder and transpose-convolute to build decoder'''

    def __init__(self, in_channel=1, n_class=2):
        self.feature_root = 64
        tf.reset_default_graph()
        self.tfx = tf.placeholder(tf.float32, [None, None, None, in_channel], 'Input')
        self.tfy = tf.placeholder(tf.float32, [None, None, None, n_class], 'Label')

        # first layer--- cov1: 7*7,64, stride2 , original:(-1,640,640,1) -> output: (-1,320,320,64)
        with tf.variable_scope('CONV1'):
            self.conv1 = tf.layers.conv2d(self.tfx, self.feature_root, (7, 7), 2, 'same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

        # second layer 3*3 maxpool and conv2_x--> output (-1,160,160,256)
        with tf.variable_scope('CONV2'):
            conv2mp = tf.layers.max_pooling2d(self.conv1, 3, 2, 'same')


if __name__ == '__main__':
