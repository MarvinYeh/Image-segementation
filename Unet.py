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


tf.logging.set_verbosity(tf.logging.INFO)


def BRW_block(input_node, iteration, in_channel, init_downsample=True):
    """BN-Relu-Weights(1*1,3*3,1*1) [pre-activation]"""
    x = input_node

    for n in range(iteration):
        for m in range(2):
            x = tf.layers.batch_normalization(x, axis=-1, momentum=0.99)
            x = tf.nn.relu(x)
            if n == 0 and m == 0:  # stride 2 for down-size in the first convolution of each layer
                if init_downsample is True:  # for layers other than conv2_, where downsample is not required
                    x = tf.layers.conv2d(x, in_channel, (1, 1), strides=2, padding='same',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                    shortcut = tf.layers.conv2d(input_node, in_channel * 4, (1, 1), strides=2, padding='same')
                else:
                    x = tf.layers.conv2d(x, in_channel, (1, 1), padding='same',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                    shortcut = tf.layers.conv2d(input_node, in_channel * 4, (1, 1), strides=1, padding='same')
            else:
                x = tf.layers.conv2d(x, in_channel, (1, 1), padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

            x = tf.layers.conv2d(x, in_channel, (3, 3), padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            x = tf.layers.conv2d(x, in_channel * 4, (1, 1), padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

        # add residual
        x = x + shortcut
        shortcut = x
    return x


def deconv2d_cat_BRW(input, conv_input, filter_size, in_channel):
    '''double the size by transpose convolution and decrease channel'''
    output_shape = tf.shape(conv_input)
    # transpose convolution with stride 2
    w = tf.get_variable('w', shape=[filter_size, filter_size, in_channel / 2, in_channel],
                        initializer=tf.contrib.layers.xavier_initializer())

    x = tf.nn.conv2d_transpose(input, w, tf.stack([output_shape[0], output_shape[-3], output_shape[-2], int(in_channel / 2)]),
                               strides=[1, 2, 2, 1], padding='SAME')
    # shape (-1,?,?, in/2)
    x = tf.concat([x, conv_input], -1)  # shape (-1,?,?, in)
    x = tf.layers.batch_normalization(x, axis=-1, momentum=0.99)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, in_channel / 2, 3, strides=(1, 1), padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def feed_in(img):
    '''process feed in image shape and dtype'''
    img = img[np.newaxis, ..., np.newaxis]
    img = img.astype(float)
    return img


class ResUnet(object):
    def __init__(self, in_channel=1, n_class=2):
        self.feature_root = 64

        tf.reset_default_graph()
        self.tfx = tf.placeholder(tf.float32, [None, None, None, in_channel], 'Input')
        self.tfy = tf.placeholder(tf.float32, [None, None, None, n_class], 'Label')

        # first layer--- cov1: 7*7,64, stride2 (-1,630,630,1) -> (-1,315,315,64)
        with tf.variable_scope('CONV1'):
            self.conv1 = tf.layers.conv2d(self.tfx, self.feature_root, (7, 7), 2, 'same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

        # second layer 3*3 maxpool and conv2_x--> (-1,158,158,256)
        with tf.variable_scope('CONV2'):
            conv2mp = tf.layers.max_pooling2d(self.conv1, 3, 2, 'same')
            self.conv2 = BRW_block(conv2mp, 1, self.feature_root, init_downsample=False)  # shape(-1,79,79,256)

        with tf.variable_scope('CONV3'):  # -->(-1,79,79,512) note the output will be 4*in_channel in each block
            self.conv3 = BRW_block(self.conv2, 4, self.feature_root * 2 ** 1,
                                   init_downsample=True)  # shape(-1,40,40,512)

        with tf.variable_scope('CONV4'):
            self.conv4 = BRW_block(self.conv3, 6, self.feature_root * 2 ** 2)  # shape(-1,20,20,1024)

        with tf.variable_scope('CONV5'):
            self.conv5 = BRW_block(self.conv4, 3, self.feature_root * 2 ** 3)  # shape(-1,10,10,2048)

        # up-convolution layer


        with tf.variable_scope('T_CONV4'):
            self.dconv4 = tf.layers.conv2d_transpose(self.conv5, self.feature_root * 2 ** 4, 3, strides=2,
                                                     padding='same')
            # shape(-1,?,?,1024)
            self.cat = tf.cat([self.deconv4, self.conv4], -1)  # add or convolution
            self.t_conv4 = BRW_block(self.cat, 1, self.feature_root * 2 ** 2)  # shape(-1,?,?,1024)

        with tf.variable_scope('T_CONV3'):
            self.dconv3 = tf.layers.conv2d_transpose(self.t_conv4, self.feature_root * 2 ** 3, 3, strides=2,
                                                     padding='same')
            # shape(-1,?,?,512)
            self.cat = tf.cat([self.deconv3, self.conv3], -1)  # add or convolution
            self.t_conv3 = BRW_block(self.cat, 1, self.feature_root * 2 ** 1)  # shape(-1,?,?,512)

        with tf.variable_scope('T_CONV2'):
            self.dconv2 = tf.layers.conv2d_transpose(self.t_conv3, self.feature_root * 2 ** 2, 3, strides=2,
                                                     padding='same')
            # shape(-1,?,?,256)
            self.cat = tf.cat([self.deconv2, self.conv2], -1)  # add or convolution
            self.t_conv2 = BRW_block(self.cat, 1, self.feature_root * 2 ** 1)  # shape(-1,?,?,256)

        with tf.variable_scope('T_CONV1'):
            self.dconv1 = tf.layers.conv2d_transpose(self.t_conv2, self.feature_root * 2 ** 2, 3, strides=2,
                                                     padding='same')
            # shape(-1,?,?,256)
            self.cat = tf.cat([self.deconv1, self.conv1], -1)  # add or convolution
            # self.t_conv4 = BRW_block(self.cat, 1, self.feature_root * 2 ** 1)  # shape(-1,?,?,256)

        self.output = self.conv3

    def train(self, img):
        img = img[np.newaxis, ..., np.newaxis]
        img = img.astype(float)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        output = self.sess.run(self.output, {self.tfx: img})

        return output



if __name__ == '__main__':
    plt.rcParams['image.cmap'] = 'gray'
    img = mpimg.imread('brain.jpeg')
    img = img[:-2, :-2]
    unet = ResUnet()
    out = unet.train(img)
    dec = unet.sess.run(unet.t_conv2, {unet.tfx: feed_in(img)})
    # conv2, conv3, conv4, conv5 = unet.sess.run([unet.conv2, unet.conv3, unet.conv4, unet.conv5],
    #                                            {unet.tfx: feed_in(img)})



    # for n in range(10):
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(conv2[0, ..., n])
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(conv3[0, ..., n])
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(conv4[0, ..., n])
    #     plt.subplot(1, 4, 4)
    #     plt.imshow(conv5[0, ..., n])

    print('hello')

    # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(unet.l1.name)[0] + '/kernel:0')

    # plt.imshow(out[])
    # plt.show()

#
# [n.name for n in tf.get_default_graph().as_graph_def().node]
#

plt.subplot(121)
plt.imshow(out[0, ..., 256])
plt.subplot(122)
plt.imshow(dec[0, ..., 0])
