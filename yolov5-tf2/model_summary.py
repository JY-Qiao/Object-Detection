# 查看模型网络结构
from yolo5 import yolo_body
from tensorflow.keras.layers import Input
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

if __name__ == '__main__':
    model_input = Input(shape=(640, 640, 3))
    num_anchors = 3
    num_classes = 80

    model = yolo_body(model_input, num_anchors, num_classes, 'x')

    model.summary()
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

# yolo_head_20: (None, 20, 20, 225)
# yolo_head_40: (None, 40, 40, 225)
# yolo_head_80: (None, 80, 80, 225)