import numpy as np
import tensorflow.keras.backend as K

from nets.utils import get_Mosaic_data


def smooth_labels(y_true, label_smoothing):
    num_classes = K.shape(y_true)[-1],
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return np.array(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)

def get_near_points(ctr_x, ctr_y, x, y):
    # ctr_x: 真实框中心点在对应卷积图上的横坐标
    # ctr_y: 真实框中心点在对应卷积图上的横坐标
    # x: 真实框中心点所在网格点的横坐标
    # y: 真实框中心点所在网格点的纵坐标
    sub_x = ctr_x - x
    sub_y = ctr_y - y
    ####################################################################################################################
    #                                               (0, -1)
    #                     (-1, 0)                   (0,  0)                            (0, 1)
    #                                               (0,  1)
    ####################################################################################################################
    if sub_x > 0.5 and sub_y > 0.5:
        return [[0, 0], [1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y > 0.5:
        return [[0, 0], [-1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y < 0.5:
        return [[0, 0], [-1, 0], [0, -1]]
    else:
        return [[0, 0], [1, 0], [0, -1]]

def get_y_trues(boxes, anchors, box_shape, head_shape, input_shape, anchors_mask, classes, label_dataset):

    threshold = 4

    # 初始化y_trues为全0数组，依次为小中大尺寸,y_trues.shape = [3, head_shape[i], head_shape[i], box_shape[1], 5+len(classes)]
    y_trues = [np.zeros((head_shape[i], head_shape[i], box_shape[1], 5 + len(classes))) for i in range(box_shape[0])]

    # boxes = [num_boxes, xmin, ymin, xmax, ymax, class_id]
    boxes = np.array(boxes, dtype='float32')
    # input_shape = (640, 640, 3)
    input_shape = np.array(input_shape, dtype='int')
    # num_layers = 3
    num_layers = box_shape[0]

    # box_best_ratios.shape = [head_shape[i], head_shape[i], box_shape[1]]
    box_best_ratios = [np.zeros((head_shape[i], head_shape[i], box_shape[1]), dtype='float32') for i in range(num_layers)]

    # 先验框形状：(3, 3, 2) --> (9, 2)
    anchors = np.reshape(anchors, (9, 2))

    # 计算boxes中每个box的中心点坐标以及box的长和宽（对应尺寸大小为640）
    boxes_xy = (boxes[..., 0:2] + boxes[..., 2:4]) // 2
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]

    # 真实框归一化处理，并将boxes中的元素替换为x, y, w, h
    boxes[..., 0:2] = boxes_xy / input_shape[0]
    boxes[..., 2:4] = boxes_wh / input_shape[1]

    # 计算真实框的宽高与先验框的宽高的比值
    ratios_of_gt_anchors = np.expand_dims(boxes_wh, 1) / np.expand_dims(anchors, 0)
    # 计算先验框的宽高与真实框的宽高的比值
    ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(boxes_wh, 1)
    # 将两种比值进行拼接
    total_ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
    # 获得先验框与真实框以及真实框与先验框的长比值和高比值中的最大值
    max_ratios = np.max(total_ratios, axis=-1)
    # idx表示预测框的索引值，ratio表示该预测框与每个先验框的长比值（互相比）和宽比值（互相比）之间的最大值
    for idx, ratio in enumerate(max_ratios):
        keep_anchors_mask = ratio < threshold
        # 找到每个真实框所属的特征层
        for layer in range(num_layers):
            # k表示网格点中第k个预测框, n表示先验框的编号
            for k, n in enumerate(anchors_mask[layer]):
            # 过滤掉比值不符合要求的先验框
                if not keep_anchors_mask[n]:
                    continue

                # 计算中心点所在网格的左上方的网格点坐标
                x = np.floor(boxes[idx, 0] * head_shape[layer]).astype('int32')
                y = np.floor(boxes[idx, 1] * head_shape[layer]).astype('int32')

                # 计算中心点最相邻的三个网格的左上方网格点
                offsets = get_near_points(boxes[idx, 0] * head_shape[layer], boxes[idx, 1] * head_shape[layer], x, y)
                # 计算中心点最相邻的三个网格的左上方网格点坐标
                for offset in offsets:
                    x = x + offset[0]
                    y = y + offset[1]

                    # 将不存在的网格点去掉
                    if x >= head_shape[layer] or x < 0 or y >= head_shape[layer] or y < 0:
                        continue

                    # 判断box_best_ratios中对应的网格点是否有值，有的话，判断其ratio大小是否小于此时的ratio值，若大于，则将其进行更新
                    # box_best_ratios.shape = [3, head_shape[i], head_shape[i], box_shape[1]]
                    if box_best_ratios[layer][x, y, k] != 0:
                        if box_best_ratios[layer][x, y, k] > ratio[n]:
                            # 重置y_trues中对应位置的真实框
                            y_trues[layer][x, y, k, :] = 0
                        else:
                            continue
                    cls = boxes[idx, 4].astype('int32')

                    y_trues[layer][x, y, k, 0:4] = boxes[idx, 0:4]
                    y_trues[layer][x, y, k, 4] = 1
                    y_trues[layer][x, y, k, 5+cls] = 1
                    box_best_ratios[layer][x, y, k] = ratio[n]

                    # 平滑标签
                    label_smoothing = 0.01
                    y_trues[layer] = smooth_labels(y_trues[layer], label_smoothing)

    label_dataset[0].append(y_trues[0])
    label_dataset[1].append(y_trues[1])
    label_dataset[2].append(y_trues[2])

    return label_dataset


def data_generator(annotations, head_shape, box_shape, classes, anchors, input_size, batch_size):
    begin = 0
    input_shape = (input_size, input_size, 3)
    anchors_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # 数据集个数必须是4的倍数，才能进行Mosaic数据增强处理
    if len(annotations) % 4 != 0:
        annotations = annotations[: len(annotations) - len(annotations) % 4]
    num = len(annotations)

    while True:
        # 每隔一个epoch打乱一次顺序
        if begin == 0:
            index = np.random.permutation(len(annotations))
            annotations = np.array(annotations)[index]
        label_dataset = [[], [], []]
        image_dataset = []
        end = begin + batch_size * 4
        times = (end - begin) // 4
        for time in range(times):
            image, boxes = get_Mosaic_data(annotations[begin + time * 4 : begin + (time + 1) * 4], (input_size, input_size))
            label_dataset = get_y_trues(boxes, anchors, box_shape, head_shape, input_shape, anchors_mask, classes, label_dataset)
            image_dataset.append(image)
        image_dataset = np.array(image_dataset)
        label_dataset = [np.array(label_dataset[0]), np.array(label_dataset[1]), np.array(label_dataset[2])]

        begin = end
        if begin + batch_size * 4 > num:
            begin = 0

        yield [image_dataset, *label_dataset], np.zeros(batch_size)
