import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import math

def get_y_pre(y_pres):
    # y_pres.shape = [batch_size,feature_size,feature_size,3,classes+location+confidence]
    # 生成一个[1,feature_size,feature_size,1,1]的张量x
    y_pres = tf.cast(y_pres, tf.float32)
    x = tf.cast(tf.reshape(tf.tile(tf.range(y_pres.shape[1]), [y_pres.shape[2]]), (1, y_pres.shape[1], y_pres.shape[2], 1, 1)), tf.float32)
    # 将张量x变成网格点的编号张量y[1,feature_size,feature_size,1,1]
    y = tf.transpose(x, (0,2,1,3,4))
    # 将张量x和y拼接在一起，并生成形状为[1,feature_size,feature_size,3,location+classes+confidence,2]的xy，也就是网格点的坐标
    xy = tf.tile(tf.concat([x,y],-1), [1, 1, 1, y_pres.shape[3], 1])
    # 将相对于网格点坐标的位移与相应的网格点坐标相加，获得预测框中心点的xy坐标
    pre_xy = tf.cast(K.sigmoid(y_pres[...,:2]), tf.float32) + xy
    # 获得预测框的置信度
    pre_con = tf.cast(K.sigmoid(y_pres[...,4:5]), tf.float32)
    # 获得预测框的class
    pre_cla = y_pres[...,5:]
    # 最后将各部分拼接在一起，形成最终形式的y_pres，形状为[batch_size,feature_size,feature_size,(cx,cy,w,h,classes,confidence),grid_coordinate]
    y_pres =  tf.concat([pre_xy,y_pres[...,2:4],pre_con,pre_cla],axis=-1)

    return y_pres

def get_IOU(y_trues,y_pres,input_size,thresh,anchors,mask):
    ratio = input_size / int(y_trues.shape[1])
    # 每一个head中的anchor形状为[3,2]，resize成[1,1,1,3,2]
    anchor_grid = K.constant(anchors, dtype='float32', shape=[1,1,1,3,2])
    # wh_true.shape = [batch,head_shape[i],head_shape[i],3,2]
    xy_true = y_trues[..., :2] * ratio  # 计算真实框的中心点坐标
    wh_true = y_trues[..., 2:4] * anchor_grid  # 计算真实框的宽高
    min_true = xy_true - wh_true / 2  # 计算真实框左下角处的坐标
    max_true = xy_true + wh_true / 2  # 计算真实框右上角处的坐标

    # 计算预测框的相关信息
    xy_pre = y_pres[..., :2] * ratio  # 计算预测框的中心点
    wh_pre = y_pres[..., 2:4] * anchor_grid  # 计算预测框的宽高
    min_pre = xy_pre - wh_pre / 2  # 计算预测框左下角处的坐标
    max_pre = xy_pre + wh_pre / 2  # 计算预测框右上角处的坐标

    # 真实框与预测框交叠处的最小坐标及最大坐标
    intersect_min = K.maximum(min_pre,min_true)
    intersect_max = K.minimum(max_pre,max_true)

    # 真实框与预测框交叠处的面积
    intersect_wh = K.maximum(intersect_max-intersect_min, 0.)
    intersect_area = intersect_wh[...,0:1] * intersect_wh[...,1:]

    true_area = wh_true[...,0:1] * wh_true[...,1:]
    pred_area = wh_pre[...,0:1] * wh_true[...,1:]

    union_area = pred_area + true_area - intersect_area
    IOU = intersect_area / union_area

    # 将IOU大于阈值的预测框con设置为1，小于阈值的输出设置为0
    if mask == 'object_mask':
        return tf.cast(IOU>thresh,tf.float32)
    elif mask == 'ignore_mask':
        return tf.cast(IOU<=thresh,tf.float32)

def ciou(box1, box2):
    # box1.shape = [batch_size, head_shape, head_shape, anchor_num, 4]
    # box2.shape = [batch_size, head_shape, head_shape, anchor_num, 4]
    # 计算预测框左上角和右下角
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_wh_half = box1_wh / 2.
    box1_mins = box1_xy - box1_wh_half  # 计算右下角坐标
    box1_maxes = box1_xy + box1_wh_half  # 计算左上角坐标
    # 计算真实框左上角和右下角
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_wh_half = box2_wh / 2.
    box2_mins = box2_xy - box2_wh_half  # 计算右下角坐标
    box2_maxes = box2_xy + box2_wh_half  # 计算左上角坐标

    # 求所有的真实框与预测框的IOU
    intersect_mins = K.maximum(box1_mins, box2_mins)
    intersect_maxes = K.minimum(box1_maxes, box2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    # 计算交叠区域面积
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    # 计算真实框与预测框自身区域面积
    box1_areas = box1_wh[..., 0] * box1_wh[..., 1]
    box2_areas = box2_wh[..., 0] * box2_wh[..., 1]
    # 计算IOU
    union_areas = box1_areas + box2_areas - intersect_areas
    iou = union_areas / K.maximum(union_areas, K.epsilon())
    # 计算中心的距离
    ctr_dis = K.sum(K.square(box1_xy - box2_xy), axis=-1)
    # 找到能够包围box1和box2的最小框的左上角和右下角
    enclose_mins = K.minimum(box1_mins, box2_mins)
    enclose_maxes = K.maximum(box1_maxes, box2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.)
    # 计算对角线距离
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # 计算参数v
    v = 4 * K.square(tf.math.atan2(box1_wh[..., 0], K.maximum(box1_wh[..., 1], K.epsilon())) - tf.math.atan2(box2_wh[..., 0], K.maximum(box2_wh[..., 1], K.epsilon()))) / (math.pi * math.pi)
    # 计算参数alpha
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    # 计算ciou
    ciou = iou - 1.0 * (ctr_dis) / K.maximum(enclose_diagonal, K.epsilon()) - alpha * v

    return ciou

def get_loss_box(y_trues,y_pres,box_scale,object_mask):
    loss_xy = K.sum(box_scale * object_mask * K.square(y_trues[...,:2]-y_pres[...,:2]))
    loss_wh = K.sum(0.5 * box_scale * object_mask * K.square(y_trues[...,2:4]-y_pres[...,2:4]))

    return loss_xy, loss_wh

def get_loss_con(y_trues,y_pres,unobj_scale,object_mask,ignore_mask):
    loss_con = K.sum(object_mask*K.square(y_pres-y_trues)+unobj_scale*ignore_mask*(1-object_mask)*y_pres)

    return loss_con

def get_loss_class(y_trues, y_pres, object_mask):
    loss_bce = K.sum(object_mask * K.binary_crossentropy(y_pres, y_trues, from_logits=True))

    return loss_bce

def get_loss_ciou(y_trues, y_pres):
    CIOU = ciou(y_trues, y_pres)
    loss_ciou = K.sum(1 - CIOU)

    return loss_ciou

def lossCalculator(y_trues,y_pres,anchors,input_size,object_scale,unobj_scale,thresh):
    y_pres = K.reshape(y_pres, shape=[-1,y_pres.shape[-3],y_pres.shape[-2],anchors.shape[0],y_pres.shape[-1]//anchors.shape[0]])
    # y_trues = K.reshape(y_trues, shape=[-1,y_pres[1],y_pres[2],y_pres[3],y_pres[4]])

    y_pres = get_y_pre(y_pres)
    # 取出真实框的confidence，值为1,形状为[batch_size,feature_size,feature_size,3,confidence]
    object_mask = get_IOU(y_trues[...,:4],y_pres[...,:4],input_size,thresh,anchors,'object_mask')
    ignore_mask = get_IOU(y_trues[...,:4],y_pres[...,:4],input_size,thresh,anchors,'ignore_mask')

    # 格式转换
    y_trues = tf.cast(y_trues, dtype=tf.float32)
    y_pres = tf.cast(y_pres, dtype=tf.float32)
    object_mask = tf.cast(object_mask, dtype=tf.float32)
    ignore_mask = tf.cast(ignore_mask, dtype=tf.float32)

    loss_xy,loss_wh = get_loss_box(y_trues[...,:4],y_pres[...,:4],object_scale,object_mask)
    loss_con = get_loss_con(y_trues[...,4:5],y_pres[...,4:5],unobj_scale,object_mask,ignore_mask)
    loss_class = get_loss_class(y_trues[...,5:],y_pres[...,5:],object_mask)
    loss_ciou = get_loss_ciou(y_trues[...,:4],y_pres[...,:4])

    loss = loss_xy + loss_wh + loss_con + loss_class + loss_ciou

    return loss, loss_xy, loss_wh, loss_con, loss_class, loss_ciou

def yolo_loss(loss_input):
    # y_trues.shape = [None,20,20,3,85] or [None,40,40,3,85] or [None,80,80,3,85]
    # y_pres.shape = [None,20,20,255] or [None,40,40,255] or [None,80,80,255]

    thresh = 0.5  # IOU阈值
    unobj_scale = 0.5 # 没有目标的预测框的权重
    object_scale = 0.5  # 有目标的预测框的权重
    input_size = 640

    anchors = np.array([[[116,90], [156,198], [373,326]],
                        [[30,61], [62,45], [59,119]],
                        [[10,13], [16,30], [33,23]]])

    y_pres = []
    y_trues = []

    num_layers = 3
    for num in range(num_layers):
        y_pre = loss_input[num]
        y_pres.append(y_pre)
        y_true = loss_input[3 + num]
        y_trues.append(y_true)

    loss_20, loss_xy_20, loss_wh_20, loss_con_20, loss_class_20, loss_ciou_20 = \
        lossCalculator(y_trues[0], y_pres[0], anchors[0], input_size, object_scale, unobj_scale, thresh)

    loss_40, loss_xy_40, loss_wh_40, loss_con_40, loss_class_40, loss_ciou_40 = \
        lossCalculator(y_trues[1], y_pres[1], anchors[1], input_size, object_scale, unobj_scale, thresh)

    loss_80, loss_xy_80, loss_wh_80, loss_con_80, loss_class_80, loss_ciou_80 = \
        lossCalculator(y_trues[2], y_pres[2], anchors[2], input_size, object_scale, unobj_scale, thresh)

    loss_total = loss_20 + loss_40 + loss_80

    return loss_total

#######################################################测试用代码#########################################################
# y_trues = [np.ones((1,20,20,3,85)), np.ones((1,40,40,3,85)), np.ones((1,80,80,3,85))]
# y_pres = [np.ones((1,20,20,255)) * 2, np.ones((1,40,40,255)) * 2, np.ones((1,80,80,255)) * 2]
# inputs = [*y_pres, *y_trues]
# print(yolo_loss(inputs))