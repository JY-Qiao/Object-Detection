import numpy as np
from get_xml import PascalVocXmlParser
import tensorflow as tf


def load_preprocessing_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = tf.cast(img, tf.float32)/ 127.5 - 1
    return img

def get_parse(fname, input_size):
    parser = PascalVocXmlParser()
    img_name = parser.get_fname(fname)
    width = parser.get_width(fname)
    height = parser.get_height(fname)
    labels = parser.get_labels(fname)
    boxes = parser.get_boxes(fname)

    for i in range(len(boxes)):
        boxes[i][0] = boxes[i][0] / width * input_size  # xmin
        boxes[i][1] = boxes[i][1] / width * input_size  # xmax
        boxes[i][2] = boxes[i][2] / height * input_size  # ymin
        boxes[i][3] = boxes[i][3] / height * input_size  # ymax

    return img_name, labels, boxes

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
        ratio = head_shape[layer_idx - 1] / input_size

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
        new_box[2] = np.log(max(w,1) / anchors[layer_idx-1][box_idx-1][0])
        new_box[3] = np.log(max(h,1) / anchors[layer_idx-1][box_idx-1][1])

        # 更新y_trues
        y_trues[layer_idx-1][x,y,box_idx-1,0:4] = new_box[0:4]  # 更新xmin,xmax,ymin,ymax
        y_trues[layer_idx-1][x,y,box_idx-1,4:5] = 1 # 更新confidence
        y_trues[layer_idx-1][x,y,box_idx-1,5+c:6+c] = 1  # 更新label

        return y_trues

def generator(imgs_path, xmls_path, head_shape, box_shape, classes, anchors, input_size):

    # 按照顺序调整好数据排列，并划分训练集和验证集
    imgs_path.sort(key=lambda img_path: img_path.split('\\')[-1].split('.jpg')[0])
    xmls_path.sort(key=lambda xml_path: xml_path.split('\\')[-1].split('.jpg')[0])

    # 初始化y_trues为全0数组，依次为小中大尺寸,.shape = [head_shape[i], head_shape[i], box_shape[1], 5+len(classes)]
    y_trues = [np.zeros((head_shape[i],head_shape[i],box_shape[1],5+len(classes))) for i in range(box_shape[0])]
    # 随机打乱
    index = np.random.permutation(len(imgs_path))
    imgs_path = np.array(imgs_path)[index]
    xmls_path = np.array(xmls_path)[index]

    dataset_image = tf.data.Dataset.from_tensor_slices(imgs_path)

    # 加载图片，生成图片数据集
    image_dataset = dataset_image.map(load_preprocessing_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    label_dataset = [[],[],[]]
    for xmls_path in xmls_path:
        fname, labels, boxes = get_parse(xmls_path, input_size)
        # 生成标签数据集
        y_trues = get_y_trues(boxes,anchors,box_shape,head_shape,input_size,classes,labels,y_trues)
        # 加载标签
        label_dataset[0].append(y_trues[0])  # shape = [len(xmls_path), [[13,13,3,85], [26,26,3,85], [52,52,3,85]]]
        label_dataset[1].append(y_trues[1])
        label_dataset[2].append(y_trues[2])
    label_dataset = tf.data.Dataset.from_tensor_slices((label_dataset[0],label_dataset[1],label_dataset[2]))

    # 生成数据集
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    train_count = int(len(imgs_path) * 0.8)
    test_count = len(imgs_path) - train_count

    return dataset, train_count, test_count


#############################################################暂时不用####################################################
# 测试用代码
# import os
# import glob
# imgs_path = os.path.join('VOCdevkit','VOC2007','JPEGImages','*.jpg')
# imgs_path = glob.glob(imgs_path)
#
# xmls_path = os.path.join('VOCdevkit','VOC2007','Annotations','*.xml')
# xmls_path = glob.glob(xmls_path)
#
# head_shape = [13,26,52]
# box_shape = [3,3]
#
# def load_coconames(file_name):
#     with open(file_name) as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#
#     return class_names
# classes = load_coconames('coco_classes.txt')
# anchors = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]
# input_size = 416
#
# dataset, train_count, test_count = generator(imgs_path, xmls_path, head_shape, box_shape, classes, anchors, input_size)
#
# print(dataset)
# print(train_count)
# print(test_count)