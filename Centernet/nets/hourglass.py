import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D, Input, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal


def conv2d(x, k, out_dim, name, stride=1):
    padding = (k - 1) // 2
    x = ZeroPadding2D(padding=padding, name=name + '.pad')(x)
    x = Conv2D(out_dim, k, strides=stride, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name=name + '.conv')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn')(x)
    x = Activation('relu', name=name + '.relu')(x)
    return x

def residual(x, out_dim, name, stride=1):

    shortcut = x
    num_channels = K.int_shape(shortcut)[-1]

    x = ZeroPadding2D(padding=1, name=name + '.pad1')(x)
    x = Conv2D(out_dim, 3, strides=stride, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name=name + '.conv1')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn1')(x)
    x = Activation('relu', name=name + '.relu1')(x)

    x = Conv2D(out_dim, 3, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name=name + '.conv2')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn2')(x)

    if num_channels != out_dim or stride != 1:
        shortcut = Conv2D(out_dim, 1, strides=stride, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name=name + '.shortcut.0')(
            shortcut)
        shortcut = BatchNormalization(epsilon=1e-5, name=name + '.shortcut.1')(shortcut)

    x = Add(name=name + '.add')([x, shortcut])
    x = Activation('relu', name=name + '.relu')(x)
    return x

def bottleneck_layer(x, num_channels, hgid):

    pow_str = 'center.' * 5
    x = residual(x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    return x

def connect_left_right(left, right, num_channels, num_channels_next, name):

    left = residual(left, num_channels_next, name=name + 'skip.0')
    left = residual(left, num_channels_next, name=name + 'skip.1')

    out = residual(right, num_channels, name=name + 'out.0')
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = UpSampling2D(name=name + 'out.upsampleNN')(out)

    out = Add(name=name + 'out.add')([left, out])
    return out    

def pre(x, num_channels):
    x = conv2d(x, 7, 128, name='pre.0', stride=2)
    x = residual(x, num_channels, name='pre.1', stride=2)
    return x

def left_features(bottom, hgid, dims):
    features = [bottom]
    for kk, nh in enumerate(dims):
        x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, str(kk)), stride=2)
        x = residual(x, nh, name='kps.%d%s.down.1' % (hgid, str(kk)))
        features.append(x)
    return features

def right_features(leftfeatures, hgid, dims):
    rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
    for kk in reversed(range(len(dims))):
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
    return rf


def create_heads(num_classes, rf1, hgid):
    y1 = Conv2D(256, 3, kernel_initializer=RandomNormal(stddev=0.02), use_bias=True, padding='same', name='hm.%d.0.conv' % hgid)(rf1)
    y1 = Activation('relu', name='hm.%d.0.relu' % hgid)(y1)
    y1 = Conv2D(num_classes, 1, use_bias=True, name='hm.%d.1' % hgid, activation = "sigmoid")(y1)

    y2 = Conv2D(256, 3, kernel_initializer=RandomNormal(stddev=0.02), use_bias=True, padding='same', name='wh.%d.0.conv' % hgid)(rf1)
    y2 = Activation('relu', name='wh.%d.0.relu' % hgid)(y2)
    y2 = Conv2D(2, 1, use_bias=True, name='wh.%d.1' % hgid)(y2)

    y3 = Conv2D(256, 3, kernel_initializer=RandomNormal(stddev=0.02), use_bias=True, padding='same', name='reg.%d.0.conv' % hgid)(rf1)
    y3 = Activation('relu', name='reg.%d.0.relu' % hgid)(y3)
    y3 = Conv2D(2, 1, use_bias=True, name='reg.%d.1' % hgid)(y3)

    return [y1,y2,y3]

def hourglass_module(num_classes, bottom, cnv_dim, hgid, dims):
    lfs = left_features(bottom, hgid, dims)

    rf1 = right_features(lfs, hgid, dims)
    rf1 = conv2d(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)

    heads = create_heads(num_classes, rf1, hgid)
    return heads, rf1


def HourglassNetwork(inpnuts, num_stacks, num_classes, cnv_dim=256, dims=[256, 384, 384, 384, 512]):
    inter = pre(inpnuts, cnv_dim)
    outputs = []
    for i in range(num_stacks):
        prev_inter = inter
        _heads, inter = hourglass_module(num_classes, inter, cnv_dim, i, dims)
        outputs.append(_heads)
        if i < num_stacks - 1:
            inter_ = Conv2D(cnv_dim, 1, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name='inter_.%d.0' % i)(prev_inter)
            inter_ = BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

            cnv_ = Conv2D(cnv_dim, 1, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False, name='cnv_.%d.0' % i)(inter)
            cnv_ = BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

            inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)
    return outputs
    
if __name__ == "__main__":
    image_input = Input(shape=(512, 512, 3))
    outputs = HourglassNetwork(image_input,2,20)
    model = Model(image_input,outputs[-1])
    model.summary()
