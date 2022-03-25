from tensorflow import keras
from tensorflow.keras.layers import (BatchNormalization, Concatenate,
                                     LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

from nets.body import DarknetConv2D, CSPDarknet53


#############################################yolo_body部分###############################################################
def DarknetConv2D_BN_Leaky(x, *args, **kwargs):
    new_kwargs = {'use_bias': False}
    new_kwargs.update(kwargs)

    x = DarknetConv2D(*args, **new_kwargs)(x)
    x = BatchNormalization()(x)
    y = LeakyReLU(alpha=0.1)(x)

    return y

def CBL3(x):
    # 进行三次卷积
    x = DarknetConv2D_BN_Leaky(x, 512, (1, 1))
    x = DarknetConv2D_BN_Leaky(x, 1024, (3, 3))
    y = DarknetConv2D_BN_Leaky(x, 512, (1, 1))

    return y

def CBL5(x, num_filters):
    # 进行五次卷积
    x = DarknetConv2D_BN_Leaky(x, num_filters, (1,1))
    x = DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
    x = DarknetConv2D_BN_Leaky(x, num_filters, (1,1))
    x = DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
    y = DarknetConv2D_BN_Leaky(x, num_filters, (1,1))

    return y

def yolo_body(inputs, num_anchors, num_classes):
    feat1, feat2, feat3 = CSPDarknet53(inputs)
    ###############################################先经过SPP网络#########################################################
    # 1.三次CBL卷积, -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x = CBL3(feat3)
    # 2.三次不同的maxpooling -> 13,13,512
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
    # 3.整合 -> 13,13,2048
    x = Concatenate()([maxpool1, maxpool2, maxpool3, x])
    # 4.三次CBL卷积 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    feat3 = CBL3(x)
    ##############################################再经过PAnet网络#########################################################
    # feat3 CBL卷积+上采样 -> 13,13,256 -> 26,26,256
    feat3_Upsamle = DarknetConv2D_BN_Leaky(feat3, 256, (1,1))
    feat3_Upsamle = UpSampling2D(2)(feat3_Upsamle)
    # feat2 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    feat2_conv = DarknetConv2D_BN_Leaky(feat2, 256, (1,1))
    feat2 = Concatenate()([feat2_conv, feat3_Upsamle])
    feat2 = CBL5(feat2, 256)
    # feat2 CBL卷积+上采样 -> 26,26,128 -> 52,52,128
    feat2_Upsample = DarknetConv2D_BN_Leaky(feat2, 128, (1,1))
    feat2_Upsample = UpSampling2D(2)(feat2_Upsample)
    # feat1 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    feat1_conv = DarknetConv2D_BN_Leaky(feat1, 128, (1,1))
    feat1 = Concatenate()([feat1_conv, feat2_Upsample])
    feat1 = CBL5(feat1, 128)
    # 第一个特征层输出
    feat1_output = DarknetConv2D_BN_Leaky(feat1, 256, (3,3))
    feat1_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(feat1_output)

    # feat1下采样 -> 26,26,256
    feat1_Downsample = ZeroPadding2D(((1,0), (1,0)))(feat1)
    feat1_Downsample = DarknetConv2D_BN_Leaky(feat1_Downsample, 256, (3,3), strides=(2,2))
    # -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    feat2 = Concatenate()([feat1_Downsample, feat2])
    feat2 = CBL5(feat2, 256)
    # 第二个特征层输出
    feat2_output = DarknetConv2D_BN_Leaky(feat2, 512, (3,3))
    feat2_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(feat2_output)

    # feat2 下采样 -> 13,13,512
    feat2_Downsample = ZeroPadding2D(((1,0), (1,0)))(feat2)
    feat2_Downsample = DarknetConv2D_BN_Leaky(feat2_Downsample, 512, (3,3), strides=(2,2))
    # -> 512,512,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    feat3 = Concatenate()([feat2_Downsample, feat3])
    feat3 = CBL5(feat3, 512)
    # 第三个特征层输出
    feat3_output = DarknetConv2D_BN_Leaky(feat3, 1024, (3,3))
    feat3_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(feat3_output)
    model = Model(inputs=inputs, outputs=[feat3_output, feat2_output, feat1_output])

    return model