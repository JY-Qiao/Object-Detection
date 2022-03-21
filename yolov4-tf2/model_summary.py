# 查看模型网络结构
from yolo import yolo_body

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

if __name__ == '__main__':
    model_input = tf.keras.Input(shape=(416, 416, 3))
    num_ancors = 3
    num_classes = 20

    model = yolo_body(model_input, num_ancors, num_classes)

    model.summary()
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

# conv2d_109: (None, 13, 13, 75)
# conv2d_101: (None, 26, 26, 75)
# conv2d_93: (None, 52, 52, 75)