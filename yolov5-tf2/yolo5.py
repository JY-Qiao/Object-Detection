from tensorflow.keras.layers import (Concatenate, UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

from nets.yolo_body import (Res_Block, DarknetConv2D, DarknetConv2D_BN_SiLU, darknet_body)


def yolo_body(inputs, num_anchors, num_classes, size):
    depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
    width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
    depth, width = depth_dict[size], width_dict[size]
    # 设定卷积的基础通道数和残差块的卷积次数
    base_channels = int(width * 64)  # 64
    base_depth = max(round(depth * 3), 1)  # 3
    # 读取Darknet处理结果
    feat1, feat2, feat3 = darknet_body(inputs, base_channels, base_depth)
    # yolo_head
    feat3_conv = DarknetConv2D_BN_SiLU(feat3, int(base_channels * 8), (1, 1))
    feat3_upsample = UpSampling2D()(feat3_conv)
    feat3_upsample = Concatenate(axis=-1)([feat3_upsample, feat2])
    feat3_upsample = Res_Block(feat3_upsample, int(base_channels * 8), base_depth, shortcut=False)

    feat2_conv = DarknetConv2D_BN_SiLU(feat3_upsample, int(base_channels * 4), (1, 1))
    feat2_upsample = UpSampling2D()(feat2_conv)
    feat2_upsample = Concatenate(axis=-1)([feat2_upsample, feat1])
    feat1_output = Res_Block(feat2_upsample, int(base_channels * 4), base_depth, shortcut=False)

    feat1_downsample = ZeroPadding2D(((1, 0), (1, 0)))(feat1_output)
    feat1_downsample = DarknetConv2D_BN_SiLU(feat1_downsample, int(base_channels * 4), (3, 3), strides=(2, 2))
    feat1_downsample = Concatenate(axis=-1)([feat1_downsample, feat2_conv])
    feat2_output = Res_Block(feat1_downsample, int(base_channels * 8), base_depth, shortcut=False)

    feat2_downsample = ZeroPadding2D(((1, 0), (1, 0)))(feat2_output)
    feat2_downsample = DarknetConv2D_BN_SiLU(feat2_downsample, int(base_channels * 8), (3, 3), strides=(2, 2))
    feat2_downsample = Concatenate(axis=-1)([feat2_downsample, feat3_conv])
    feat3_output = Res_Block(feat2_downsample, int(base_channels * 16), base_depth, shortcut=False)

    out2 = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1), strides=(1, 1))(feat1_output)
    out1 = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1), strides=(1, 1))(feat2_output)
    out0 = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1), strides=(1, 1))(feat3_output)

    model = Model(inputs=inputs, outputs=[out0, out1, out2])

    return model
