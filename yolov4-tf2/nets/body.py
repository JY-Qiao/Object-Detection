from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, ZeroPadding2D)
from tensorflow.keras.regularizers import l2


def Mish(x):
    y = x * K.tanh(K.softplus(x))
    return y

def DarknetConv2D(*args,**kwargs):
    new_kwargs = {'kernel_regularizer':l2(5e-4), 'kernel_initializer' : RandomNormal(stddev=0.02)}
    new_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2,2) else 'same'
    new_kwargs.update(kwargs)

    return Conv2D(*args, **new_kwargs)

def DarknetConv2D_BN_Mish(x, *args, **kwargs):
    new_kwargs = {'use_bias': False}
    new_kwargs.update(kwargs)

    x = DarknetConv2D(*args, **new_kwargs)(x)
    x = BatchNormalization()(x)
    y = Mish(x)
    return y


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(preconv1, num_filters, (3, 3), strides=(2, 2))

    res_right = DarknetConv2D_BN_Mish(preconv1, num_filters // 2 if all_narrow else num_filters, (1, 1))

    res_left = DarknetConv2D_BN_Mish(preconv1, num_filters // 2 if all_narrow else num_filters, (1, 1))
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Mish(res_left, num_filters // 2, (1, 1))
        y = DarknetConv2D_BN_Mish(x, num_filters // 2 if all_narrow else num_filters, (3, 3))
        res_left = Add()([res_left, y])
    res_left = DarknetConv2D_BN_Mish(res_left, num_filters // 2 if all_narrow else num_filters, (1, 1))

    result = Concatenate()([res_left, res_right])

    return DarknetConv2D_BN_Mish(result, num_filters, (1, 1))

def CSPDarknet53(x):
    x = DarknetConv2D_BN_Mish(x, 32, (3,3))
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x1 = resblock_body(x, 256, 8)  # 输出[1,52,52,256]
    x2 = resblock_body(x1, 512, 8)  # 输出[1,26,26,512]
    x3 = resblock_body(x2, 1024, 4)  # 输出[1,13,13,1024]

    return x1, x2, x3