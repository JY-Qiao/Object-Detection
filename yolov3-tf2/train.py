import os
import glob

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

import numpy as np

from nets.yolo import yolo_body
from utils.yolo_loss import yolo_loss
from preprocessing import generator

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# 获取图片及其标注文件的路径
xmls_dir = os.path.join('VOCdevkit', 'VOC2007', 'Annotations', '*.xml')
imgs_dir = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages', '*.jpg')

xmls_path = glob.glob(xmls_dir)
imgs_path = glob.glob(imgs_dir)

# 获取COCO数据集类别文件的路径
cls_path = 'coco_classes.txt'

# 读取COCO数据集的类别
def load_coco_classes(file_path):
    with open(file_path) as f:
        class_name = f.readlines()
        classes = [c.strip() for c in class_name]
    return classes

# 设置训练参数
num_anchors = 3  # 对应一个grid中需要预测的三个框
input_size = 416  # 将输入图片的尺寸定义为416*416
batch_size = 1
classes = load_coco_classes(cls_path)
num_classes = len(classes)
anchors = np.array([[[10,13], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]])  # 设置先验框的长宽
head_shape = [13,26,52]
box_shape = [3,3]  # 3个yolo head以及每个grid中有3个box

# 建立模型
model_input = Input(shape=(416,416,3))
conv2d_58, conv2d_66, conv2d_74 = yolo_body(model_input, num_anchors, num_classes)
model = Model(inputs=model_input, outputs=[conv2d_58, conv2d_66, conv2d_74])
print('已构建模型')

# 读取训练数据集以及验证数据集
dataset,train_count, test_count = generator(imgs_path, xmls_path, head_shape, box_shape, classes, anchors, input_size)
dataset_train = dataset.skip(test_count)
dataset_test = dataset.take(test_count)

BUFFER_SIZE = 100
STEPS_PER_EPOCH = train_count // batch_size
VALIDATION_STEPS = test_count // batch_size

train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(batch_size).repeat(-1)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = dataset_test.batch(batch_size)

# 编译模型
model.compile(optimizer=Adam(lr=0.0001), loss={'conv2d_58':yolo_loss, 'conv2d_66':yolo_loss, 'conv2d_74':yolo_loss})
print('已编译模型')

EPOCHS = 100

model.fit(train_dataset,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          validation_data=test_dataset)

# 保存模型
model.save_weights('logs/yolo_v3_weights.h5')