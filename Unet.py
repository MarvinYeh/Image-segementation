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

tf.logging.set_verbosity(tf.logging.INFO)


def BRW_block(input_node, iteration, in_channel, init_downsample=True):
    """BN-Relu-Weights(1*1,3*3,1*1) [pre-activation]: https://arxiv.org/pdf/1603.05027.pdf"""
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
    '''double the size by transpose convolution and decrease channel
    after concat the transposted convolution and conv in the upper layer.
    a 3*3 conv is perform for anti-liasing'''
    x = tf.layers.conv2d_transpose(input, in_channel, filter_size, strides=2, padding='same')
    x = tf.concat([x, conv_input], -1)  # shape (-1,?,?, in)
    x = tf.layers.batch_normalization(x, axis=-1, momentum=0.99)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, in_channel / 2, 3, strides=(1, 1), padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def feed_in(img):
    '''process feed in image shape and dtype'''
    img = img[np.newaxis, ..., np.newaxis]
    img = fix_pad_image(img.astype(float))
    return img


def sample_minibatch(data,labels,batchsize,shuffle=True):
    if shuffle:
        ind = np.arange(data.shape[0])
        np.random.shuffle(ind)
    for start_idx in range(0,data.shape[0]-batchsize+1,batchsize):
        if shuffle:
            excerpt = ind[start_idx:start_idx+batchsize]
        else:
            excerpt = slice(start_idx,start_idx+batchsize)
        yield data[excerpt], labels[excerpt]
        

class ResUnet(object):
    '''resnet50 Unet, preactivation, pixelwise xentropy.(is it better to do dense focal loss?)
    3*3conv after concat the conv and transpose convolution layer for antiliasing
    softm'''

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
            self.conv2 = BRW_block(conv2mp, 1, self.feature_root,
                                   init_downsample=False)  # note the output will be 4*in_channel in each block

        with tf.variable_scope('CONV3'):  # -->(-1,80,80,512)
            self.conv3 = BRW_block(self.conv2, 4, self.feature_root * 2 ** 1,
                                   init_downsample=True)

        with tf.variable_scope('CONV4'):  # -->(-1,40,40,1024)
            self.conv4 = BRW_block(self.conv3, 6, self.feature_root * 2 ** 2)

        with tf.variable_scope('CONV5'):  # -->(-1,20,20,2048)
            self.conv5 = BRW_block(self.conv4, 3, self.feature_root * 2 ** 3)

        # up-convolution layer

        with tf.variable_scope('T_CONV4'):  # -->(-1,40,40,1024)
            self.t_conv4 = deconv2d_cat_BRW(self.conv5, self.conv4, 3, self.feature_root * 2 ** 4)
        #
        with tf.variable_scope('T_CONV3'):  # -->(-1,80,80,512)
            self.t_conv3 = deconv2d_cat_BRW(self.t_conv4, self.conv3, 3, self.feature_root * 2 ** 3)

        with tf.variable_scope('T_CONV2'):  # -->(-1,160,160,256)
            self.t_conv2 = deconv2d_cat_BRW(self.t_conv3, self.conv2, 3, self.feature_root * 2 ** 2)

        with tf.variable_scope('T_CONV1'):  # -->(-1,320,320,128)
            self.t_conv1 = deconv2d_cat_BRW(self.t_conv2, self.conv1, 3, self.feature_root * 2)

        with tf.variable_scope('Output'):
            self.x = tf.layers.conv2d_transpose(self.t_conv1, self.feature_root, 3, strides=(2, 2), padding='same')
            self.x = tf.layers.batch_normalization(self.x, axis=-1, momentum=0.99)
            self.x = tf.nn.relu(self.x)
            self.output = tf.layers.conv2d(self.x, n_class, 3, strides=(1, 1), padding='same')

    def train(self, img, optimizer='adam', loss='dice_coeff', lr=0.01):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=4)
        saver.save(unet.sess, 'ResUnet',global_step=5000,write_meta_graph=False)
        if loss == 'cross_entropy':
            '''pixel-wise weighted is calculated to prevent imbalance label category'''
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pixel_wise_softmax(self.output),
                                                                   labels=self.tfy)

        elif loss == 'dice_coeff':
            prediction = pixel_wise_softmax(self.output)
            self.loss = -2 * tf.reduce_sum(prediction * self.tfy) / (
            1e-5 + tf.reduce_sum(prediction * self.tfy) + tf.reduce_sum(self.tfy))

        if optimizer == 'adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss,global_step=tf.train.global_step())
        # output = self.sess.run(self.output, {self.tfx: img})

    def predict(self, img):
        prediction = self.sess.run(self.output,{self.tfx:img})
        return prediction


if __name__ == '__main__':
    plt.rcParams['image.cmap'] = 'gray'
    EPOCH =20
    MB =16
    LR = 0.02


    #read sample image
    img = mpimg.imread('brain.jpeg')
    '''read in dataset and divide into train and test using train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    '''


    unet = ResUnet()
    unet.train()

    loss_history=[]
    for epoch in range(EPOCH):
        for n, batch in enumerate(sample_minibatch(train_x,train_y,MB,shuffle=True)):
            b_x, b_y = batch
            _, loss =  unet.sess.run([unet.train_op, unet.loss],{unet.tfx:feed_in(img),unet.tfy:feed_in(b_y)})
            loss_history.append(loss)

            if n % 10 == 0
                v_loss = unet.sess.run(unet.loss,{unet.tfx:feed_in(test_x), unet.tfy:feed_in(test_y)})
                print('loss:%.4f| val:%.4f'.format(np.mean(loss_history),np.mean(v_loss)))

