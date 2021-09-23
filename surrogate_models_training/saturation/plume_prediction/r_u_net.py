'''
Created on May 27, 2020

@author: tang39
'''
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend


def reset_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def para_RUNet(input_shape, seed_value,
               conv_filter=[16, 32, 32, 64],
               conv_kernel=[3, 3, 3, 3],
               conv_stride=[(2, 2), (1, 1), (2, 2), (1, 1)],
               conv_reg=1e-5,
               res_num=4,
               res_kernel=3,
               res_reg=1e-5):

    reset_random_seeds(seed_value)
    # encoder
    inputs = layers.Input(shape=input_shape, name='original_img')
    enc_list = []

    for ii in range(0, len(conv_filter)):
        if ii == 0:
            x = conv_bn_relu(
                inputs, conv_filter[ii], conv_kernel[ii], stride=conv_stride[ii], reg_weights=conv_reg)
            enc_list.append(x)
        else:
            x = conv_bn_relu(
                x, conv_filter[ii], conv_kernel[ii], stride=conv_stride[ii], reg_weights=conv_reg)
            enc_list.append(x)

    for ii in range(0, res_num):
        x = res_conv(x, conv_filter[-1], res_kernel, reg_weights=res_reg)

    for ii in reversed(range(0, len(conv_filter))):
        merge = layers.Concatenate()([enc_list[ii], x])
        x = dconv_bn_relu(
            merge, conv_filter[ii], conv_kernel[ii], stride=conv_stride[ii], reg_weights=conv_reg)

    output = layers.Conv2D(
        1, conv_kernel[0], padding='same', activation='sigmoid')(x)

    output_shape = backend.int_shape(output)
    N = np.array([output_shape[1] - input_shape[0],
                  output_shape[2] - input_shape[1]])

    if (np.sum(N) > 0):
        M0 = np.array([N[0] // 2, N[0] - N[0] // 2])
        M1 = np.array([N[1] // 2, N[1] - N[1] // 2])
        output = layers.Cropping2D((M0, M1))(output)

    print('output shape is ', backend.int_shape(output))
    return tf.keras.Model(inputs=inputs, outputs=output)

def RUNet(input_shape, seed_value):

    reset_random_seeds(seed_value)
    # encoder
    inputs = layers.Input(shape=input_shape, name='original_img')
    enc1 = conv_bn_relu(inputs, 16, 3, stride=(2, 2))  # (50,50,16)
    enc2 = conv_bn_relu(enc1, 32, 3, stride=(1, 1))  # (50,50,32)
    enc3 = conv_bn_relu(enc2, 64, 3, stride=(2, 2))  # (25,25,64)
    enc4 = conv_bn_relu(enc3, 64, 3, stride=(1, 1))  # (25,25,64)
    x = res_conv(enc4, 64, 3)
    x = res_conv(x, 64, 3)
    x = res_conv(x, 64, 3)  # (25,25,64)
    # decoder

    dec4 = res_conv(x, 64, 3)  # (25,25,64)
    merge4 = layers.Concatenate()([enc4, dec4])  # (25,25,128)
    dec3 = dconv_bn_relu(merge4, 64, 3, stride=(1, 1))  # (25,25,64)
    merge3 = layers.Concatenate()([enc3, dec3])  # (25,25,128)
    dec2 = dconv_bn_relu(merge3, 64, 3, stride=(2, 2))  # (50,50,64)
    merge2 = layers.Concatenate()([enc2, dec2])  # (50,50,96)
    dec1 = dconv_bn_relu(merge2, 32, 3, stride=(1, 1))  # (50,50,32)
    merge1 = layers.Concatenate()([enc1, dec1])  # (50,50,48)
    dec0 = dconv_bn_relu(merge1, 16, 3, stride=(2, 2))  # (100,100,16)

    output = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(dec0)

    output_shape = backend.int_shape(output)
    N = np.array([output_shape[1] - input_shape[0],
                  output_shape[2] - input_shape[1]])
    print('output shape is ', output_shape)

    if (np.sum(N) > 0):
        M0 = np.array([N[0] // 2, N[0] - N[0] // 2])
        M1 = np.array([N[1] // 2, N[1] - N[1] // 2])
        output = layers.Cropping2D((M0, M1))(output)

    return tf.keras.Model(inputs=inputs, outputs=output)

def conv_bn_relu(input_data, n_filter, kernel_size, stride, reg_weights=0.00001):

    x = layers.Conv2D(n_filter, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(reg_weights))(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def res_conv(input_data, n_filter, kernel_size, stride=(1, 1), reg_weights=0.00001):

    x = layers.Conv2D(n_filter, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(reg_weights))(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(n_filter, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(reg_weights))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([input_data, x])

    return x

def dconv_bn_relu(input_data, n_filter, kernel_size, stride=(2, 2), reg_weights=0.00001):

    x = layers.UpSampling2D(size=stride)(input_data)
#    x = ReflectionPadding2D(padding=(int(nb_row/2), int(nb_col/2)))(x)
#    x = layers.Conv2D(n_filter,(nb_row, nb_col), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
    x = layers.Conv2D(n_filter, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(reg_weights))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = backend.image_data_format()

        self.padding = padding
        padding = tuple(padding)
        if len(padding) == 2:
            self.top_pad = padding[0]
            self.bottom_pad = padding[0]
            self.left_pad = padding[1]
            self.right_pad = padding[1]
        elif len(padding) == 4:
            self.top_pad = padding[0]
            self.bottom_pad = padding[1]
            self.left_pad = padding[2]
            self.right_pad = padding[3]
        else:
            raise TypeError('`padding` should be tuple of int '
                            'of length 2 or 4, or dict. '
                            'Found: ' + str(padding))
        self.dim_ordering = dim_ordering
        self.input_spec = [layers.InputSpec(ndim=4)]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering in {'channels_last', 'channels_first'}:
            rows = input_shape[1] + self.top_pad + \
                self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + \
                self.right_pad if input_shape[2] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Unknown data_format:', self.dim_ordering)

    def call(self, x, mask=None):
        top_pad = self.top_pad
        bottom_pad = self.bottom_pad
        left_pad = self.left_pad
        right_pad = self.right_pad

        paddings = [[0, 0], [top_pad, bottom_pad],
                    [left_pad, right_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
