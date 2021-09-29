import tensorflow as tf
from PIL import Image

from frcnn import FRCNN


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)

        image = image.convert("RGB")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
        r_image.save("img.jpg", quality = 95)
