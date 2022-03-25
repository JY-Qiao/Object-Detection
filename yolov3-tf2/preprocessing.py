import numpy as np
from get_xml import PascalVocXmlParser
import tensorflow as tf


def load_preprocessing_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = tf.cast(img, tf.float32)/ 127.5 - 1
    return img

def get_parse(xml_path, input_size):
    parser = PascalVocXmlParser()

    width = parser.get_width(xml_path)
    height = parser.get_height(xml_path)
    labels = parser.get_labels(xml_path)
    boxes = parser.get_boxes(xml_path)

    for i in range(len(boxes)):
        boxes[i][0] = boxes[i][0] / width * input_size  # xmin
        boxes[i][1] = boxes[i][1] / width * input_size  # xmax
        boxes[i][2] = boxes[i][2] / height * input_size  # ymin
        boxes[i][3] = boxes[i][3] / height * input_size  # ymax

    return labels, boxes

def get_IOU(true_box, anchor_box):
    width = min(true_box[1], anchor_box[1]) - true_box[0]
    height = min(true_box[3], anchor_box[3]) - true_box[2]
    intersect = width * height
    merge = (true_box[1] - true_box[0]) * (true_box[3] - true_box[2]) + (anchor_box[1] - anchor_box[0]) * (anchor_box[3] - anchor_box[2])
    IOU =intersect / (merge - intersect)

    return IOU

def get_anchor(anchors, box):
    IOU_list = []
    anchor_list = np.zeros((len(anchors[0])*len(anchors[1]), 4), dtype='float32')  # shape = [9,4]

    for i in range(len(anchors[0])):
        for j in range(len(anchors[1])):
            anchor_list[i*j][0] = box[0]  # 计算xmin
            anchor_list[i*j][1] = anchors[i][j][0] + anchor_list[i*j][0]  # 计算xmax
            anchor_list[i*j][2] = box[2]  # 计算ymin
            anchor_list[i*j][3] = anchors[i][j][1] + anchor_list[i*j][2]  # 计算ymax

            IOU = get_IOU(box, anchor_list[i*j])
            IOU_list.append(IOU)

    anchor_idx = IOU_list.index(max(IOU_list))

    return anchor_idx

def get_y_trues(boxes,anchors,box_shape,head_shape,input_size,classes,labels,y_trues):
    new_box = np.zeros((4), dtype='float32')

    for i in range(len(boxes)):
        anchor_idx = get_anchor(anchors,boxes[i])

        layer_idx = anchor_idx // box_shape[0]  # 对应第几个yolo head
        box_idx = anchor_idx % box_shape[1]  # 对应网格点中第几个框

        # 将真实框的尺寸按照特征图尺寸进行缩放
        ratio = head_shape[layer_idx] / input_size

        # 计算中心点坐标
        ctr_x = (boxes[i][0] + boxes[i][1]) / 2 * ratio
        ctr_y = (boxes[i][2] + boxes[i][3]) / 2 * ratio

        # 计算真实框的位置信息
        x = np.floor(ctr_x).astype('int32')
        y = np.floor(ctr_y).astype('int32')
        w = boxes[i][1] - boxes[i][0]
        h = boxes[i][3] - boxes[i][2]

        c = classes.index(labels[i])

        new_box[0] = ctr_x
        new_box[1] = ctr_y
        new_box[2] = np.log(max(w,1) / anchors[layer_idx][box_idx][0])
        new_box[3] = np.log(max(h,1) / anchors[layer_idx][box_idx][1])

        # 更新y_trues
        y_trues[layer_idx][x,y,box_idx,0:4] = new_box[0:4]  # 更新xmin,xmax,ymin,ymax
        y_trues[layer_idx][x,y,box_idx,4:5] = 1 # 更新confidence
        y_trues[layer_idx][x,y,box_idx,5+c:6+c] = 1  # 更新label

        return y_trues

def generator(imgs_path, xmls_path, head_shape, box_shape, classes, anchors, input_size, batch_size):
    num = len(imgs_path)
    begin = 0

    while True:
        # 每隔一个epoch打乱一次顺序
        if begin == 0:
            index = np.random.permutation(len(imgs_path))
            imgs_path = np.array(imgs_path)[index]
            xmls_path = np.array(xmls_path)[index]
        # 初始化y_trues为全0数组，依次为小中大尺寸,.shape = [head_shape[i], head_shape[i], box_shape[1], 5+len(classes)]
        y_trues = [np.zeros((head_shape[i], head_shape[i], box_shape[1], 5 + len(classes))) for i in
                   range(box_shape[0])]
        image_dataset = []
        end = begin + batch_size
        # 加载图片，生成图片数据集
        for img_path in imgs_path[begin:end]:
            image_dataset.append(load_preprocessing_image(img_path))
        image_dataset = np.array(image_dataset)
        # 生成标签数据集
        label_dataset = [[], [], []]
        for xml_path in xmls_path[begin:end]:
            labels, boxes = get_parse(xml_path, input_size)
            # 生成标签数据集
            y_trues = get_y_trues(boxes, anchors, box_shape, head_shape, input_size, classes, labels, y_trues)
            # 加载标签
            label_dataset[0].append(y_trues[0])  # shape = [len(xmls_path), [[13,13,3,85], [26,26,3,85], [52,52,3,85]]]
            label_dataset[1].append(y_trues[1])
            label_dataset[2].append(y_trues[2])
        label_dataset = [np.array(label_dataset[0]), np.array(label_dataset[1]), np.array(label_dataset[2])]

        begin = end
        if begin + batch_size > num:
            begin = 0

        yield [image_dataset, *label_dataset], np.zeros(batch_size)