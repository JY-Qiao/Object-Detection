import os

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

import numpy as np

from yolo import yolo_body
from nets.loss import yolo_loss
from nets.utils import WarmUpCosineDecayScheduler
from preprocessing import data_generator

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpu,
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# 设置文件读取路径
cls_txt_path = os.path.join('model_data', 'voc_classes.txt')
dataset_txt_path = 'voc_train.txt'
anchors_path = os.path.join('model_data', 'yolo_anchors.txt')


# 从保存好类别的txt文件中读取数据集的类别
def load_classes(file_path):
    with open(file_path) as f:
        class_name = f.readlines()
        classes = [c.strip() for c in class_name]
    return classes

# 从保存好先验框尺寸的txt文件中读取先验框的尺寸
def load_anchors(file_path):
    with open(file_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape((3, 3, 2))  # .shape = [3,3,2]


with open(dataset_txt_path) as f:
    annotations = f.readlines()

# 设置训练参数
num_anchors = 3  # 一个grid中需要预测的三个框
input_size = 416  # 将输入图片的尺寸定义为416*416
batch_size = 2
classes = load_classes(cls_txt_path)
num_classes = len(classes)
anchors = load_anchors(anchors_path)
head_shape = [13, 26, 52]
box_shape = [3,3]  # 3个yolo head以及每个grid中有3个box

# 构建模型
model_input = Input(shape=(416, 416, 3))
model = yolo_body(model_input, num_anchors, num_classes)
print('已构建模型')

# 读取训练集以及验证集
train_count = int(len(annotations) * 0.8)
val_count = len(annotations) - train_count

STEPS_PER_EPOCH = train_count // batch_size
VALIDATION_STEPS = val_count // batch_size

Init_epoch = 0  # 起始世代
Freeze_epoch = 50  # 设置训练冻结世代
EPOCHS = 100  # 总训练世代
learning_rate_base = 1e-3  # 起始学习率

freeze_layers = 249
for layer in range(freeze_layers):
    model.layers[layer].trainable = False
print('Freeze the first {} layers of total {} layers'.format(freeze_layers, len(model.layers)))

# 解冻前训练
# 预热世代
warmup_epoch = int((Freeze_epoch - Init_epoch) * 0.2)
total_steps = int((Freeze_epoch - Init_epoch) * train_count / batch_size)
warmup_steps = int(warmup_epoch * train_count / batch_size)
# 学习率
reduce_learning_rate = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base, total_steps=total_steps,
                                                            warmup_learning_rate=1e-4, warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=train_count, min_learning_rate=1e-6)

# 编译模型
model.compile(optimizer=Adam(), loss={'conv2d_109':yolo_loss, 'conv2d_101':yolo_loss, 'conv2d_93':yolo_loss})
print('已编译模型')
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_count, val_count, batch_size))
model.fit(data_generator(annotations[:train_count], head_shape, box_shape, classes, anchors, input_size, batch_size),
          epochs=Freeze_epoch,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=data_generator(annotations[train_count:], head_shape, box_shape, classes, anchors, input_size, batch_size),
          validation_steps=VALIDATION_STEPS,
          initial_epoch=Init_epoch,
          callbacks=[reduce_learning_rate])
for layer in range(freeze_layers):
    model.layers[layer].trainable = True

# 解冻后训练
# 预热世代
warmup_epoch = int((EPOCHS - Freeze_epoch) * 0.2)
total_steps = int((EPOCHS - Freeze_epoch) * train_count / batch_size)
warmup_steps = int(warmup_epoch * train_count / batch_size)
reduce_learning_rate = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base, total_steps=total_steps,
                                                  warmup_learning_rate=1e-4, warmup_steps=warmup_steps,
                                                  hold_base_rate_steps=train_count, min_learning_rate=1e-6)
# 编译模型
model.compile(optimizer=Adam(), loss={'conv2d_109':yolo_loss, 'conv2d_101':yolo_loss, 'conv2d_93':yolo_loss})
print('已编译模型')
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_count, val_count, batch_size))
model.fit(data_generator(annotations[:train_count], head_shape, box_shape, classes, anchors, input_size, batch_size),
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=data_generator(annotations[train_count:], head_shape, box_shape, classes, anchors, input_size, batch_size),
          validation_steps=VALIDATION_STEPS,
          initial_epoch=Freeze_epoch,
          callbacks=[reduce_learning_rate])


# 保存模型
model.save_weights('logs/yolo4_weights.h5')