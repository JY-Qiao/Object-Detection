from functools import reduce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Conv2D,
                                     MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def SiLU(inputs):
    y = inputs * K.sigmoid(inputs)

    return y

def Focus(inputs):
    # inputs.shape = [1, 640, 640, 3]
    y = tf.concat(
            [inputs[...,  ::2,  ::2, :],
             inputs[..., 1::2,  ::2, :],
             inputs[...,  ::2, 1::2, :],
             inputs[..., 1::2, 1::2, :]],
             axis=-1
        )

    return y

def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4), 'kernel_initializer': random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_SiLU(x, *args, **kwargs):
    new_kwargs = {'use_bias': False}
    new_kwargs.update(kwargs)

    x = compose(
        DarknetConv2D(*args, **new_kwargs),
        BatchNormalization(momentum=0.97, epsilon=0.001))(x)

    y = SiLU(x)

    return y

def Res_Unit(x, out_channels, shortcut=True):
    y = DarknetConv2D_BN_SiLU(x, out_channels, (1, 1))
    y = DarknetConv2D_BN_SiLU(y, out_channels, (3, 3))
    if shortcut:
        y = Add()([x, y])
    return y

def Res_Block(x, num_filters, num_blocks, shortcut=True, expansion=0.5):
    hidden_channels = int(num_filters * expansion)

    x_1 = DarknetConv2D_BN_SiLU(x, hidden_channels, (1, 1))
    x_2 = DarknetConv2D_BN_SiLU(x, hidden_channels, (1, 1))

    for num in range(num_blocks):
        x_1 = Res_Unit(x_1, hidden_channels, shortcut=shortcut)

    y = Concatenate()([x_1, x_2])

    return DarknetConv2D_BN_SiLU(y, num_filters, (1, 1))

def SPP(x, out_channels):
    x = DarknetConv2D_BN_SiLU(x, out_channels // 2, (1, 1))

    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])

    y = DarknetConv2D_BN_SiLU(x, out_channels, (1, 1))

    return y

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False):
    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(x, num_filters, (3, 3), strides=(2, 2))
    if last:
        x = SPP(x, num_filters)
    return Res_Block(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion)

def darknet_body(x, base_channels, base_depth):
    # 640, 640, 3 => 320, 320, 12
    x = Focus(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(x, base_channels, (3, 3))
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth)
    # 160, 160, 128 => 80, 80, 256
    feat1 = resblock_body(x, base_channels * 4, base_depth * 3)
    # 80, 80, 256 => 40, 40, 512
    feat2 = resblock_body(feat1, base_channels * 8, base_depth * 3)
    # 40, 40, 512 => 20, 20, 1024
    feat3 = resblock_body(feat2, base_channels * 16, base_depth, shortcut=False, last=True)

    return feat1, feat2, feat3