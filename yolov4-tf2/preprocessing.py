import numpy as np
import tensorflow.keras.backend as K

from nets.utils import get_Mosaic_data

def get_IOU(true_box, anchor_box):
    width = min(true_box[1], anchor_box[1]) - max(true_box[0], anchor_box[0])
    height = min(true_box[3], anchor_box[3]) - max(true_box[2], anchor_box[2])
    if width < 0 or height < 0:
        return 0
    else:
        intersect = width * height
        merge = (true_box[1] - true_box[0]) * (true_box[3] - true_box[2]) + (anchor_box[1] - anchor_box[0]) * (anchor_box[3] - anchor_box[2])
        IOU = intersect / (merge - intersect)
        return IOU

def get_anchor(anchors, xmin, ymin, xmax, ymax):
    IOU_list = []
    anchor_list = np.zeros((len(anchors[0])*len(anchors[1]), 4), dtype='float32')  # shape = [9,4]

    for i in range(len(anchors[0])):
        for j in range(len(anchors[1])):
            anchor_list[i*j][0] = xmin  # 计算xmin
            anchor_list[i*j][1] = anchors[i][j][0] + anchor_list[i*j][0]  # 计算xmax
            anchor_list[i*j][2] = ymin  # 计算ymin
            anchor_list[i*j][3] = anchors[i][j][1] + anchor_list[i*j][2]  # 计算ymax

            IOU = get_IOU([xmin, xmax, ymin, ymax], anchor_list[i*j])
            IOU_list.append(IOU)

    anchor_idx = IOU_list.index(max(IOU_list))

    return anchor_idx

def smooth_labels(y_true, label_smoothing):
    num_classes = K.shape(y_true)[-1],
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return np.array(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)

def get_y_trues(boxes, anchors, box_shape, head_shape, input_size, classes, label_dataset):
    new_box = np.zeros((5), dtype='float32')
    # 初始化y_trues为全0数组，依次为小中大尺寸,y_trues.shape = [3, head_shape[i], head_shape[i], box_shape[1], 5+len(classes)]
    y_trues = [np.zeros((head_shape[i], head_shape[i], box_shape[1], 5 + len(classes))) for i in range(box_shape[0])]

    for box in boxes:
        xmin, ymin, xmax, ymax, class_idx = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])

        anchor_idx = get_anchor(anchors, xmin, ymin, xmax, ymax)

        layer_idx = anchor_idx // box_shape[0]  # 对应第几个yolo head
        box_idx = anchor_idx % box_shape[1]  # 对应网格点中第几个框

        # 将真实框的尺寸按照特征图尺寸进行缩放
        ratio = head_shape[layer_idx] / input_size

        # 计算中心点坐标
        ctr_x = (xmin + xmax) / 2 * ratio
        ctr_y = (ymin + ymax) / 2 * ratio

        # 计算真实框的位置信息
        x = np.floor(ctr_x).astype('int32')
        y = np.floor(ctr_y).astype('int32')
        w = xmax - xmin
        h = ymax - ymin

        c = class_idx

        new_box[0] = ctr_x
        new_box[1] = ctr_y
        new_box[2] = np.log(max(w, 1) / anchors[layer_idx][box_idx][0])
        new_box[3] = np.log(max(h, 1) / anchors[layer_idx][box_idx][1])

        # 更新y_trues
        y_trues[layer_idx][x, y, box_idx, 0:4] = new_box[0:4]  # 更新x_ctr-x,y_ctr-y,np.log(width/(13 or 26 or 52)),np.log(height/(13 or 26 or 52))
        y_trues[layer_idx][x, y, box_idx, 4:5] = 1  # 更新confidence
        y_trues[layer_idx][x, y, box_idx, 5 + c:6 + c] = 1  # 更新label

        # 平滑标签
        label_smoothing = 0.01
        y_trues[layer_idx] = smooth_labels(y_trues[layer_idx], label_smoothing)

    label_dataset[0].append(y_trues[0])  # shape = [len(xmls_path), [[13,13,3,85], [26,26,3,85], [52,52,3,85]]]
    label_dataset[1].append(y_trues[1])
    label_dataset[2].append(y_trues[2])

    return label_dataset

def data_generator(annotations, head_shape, box_shape, classes, anchors, input_size, batch_size):
    begin = 0

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
            label_dataset = get_y_trues(boxes, anchors, box_shape, head_shape, input_size, classes, label_dataset)
            image_dataset.append(image)
        image_dataset = np.array(image_dataset)
        label_dataset = [np.array(label_dataset[0]), np.array(label_dataset[1]), np.array(label_dataset[2])]

        begin = end
        if begin + batch_size * 4 > num:
            begin = 0

        yield [image_dataset, *label_dataset], np.zeros(batch_size)
