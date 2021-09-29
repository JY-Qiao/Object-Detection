import copy
import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tqdm import tqdm

from frcnn import FRCNN
from nets.frcnn_training import get_new_img_size
from utils.anchors import get_anchors
from utils.utils import BBoxUtility


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class mAP_FRCNN(FRCNN):
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        f = open("./input/detection-results/"+image_id+".txt","w") 

        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]

        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)
        photo = np.array(image,dtype = np.float64)

        photo = preprocess_input(np.expand_dims(photo,0))
        rpn_pred = self.model_rpn_get_pred(photo)
        rpn_pred = [x.numpy() for x in rpn_pred]

        base_feature_width, base_feature_height = self.get_img_output_length(width, height)
        anchors = get_anchors([base_feature_width, base_feature_height], width, height)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        base_layer = rpn_pred[2]
        proposal_box = np.array(rpn_results)[:, :, 1:]
        temp_ROIs = np.zeros_like(proposal_box)
        temp_ROIs[:, :, [0, 1, 2, 3]] = proposal_box[:, :, [1, 0, 3, 2]]
        classifier_pred = self.model_classifier_get_pred([base_layer, temp_ROIs])
        classifier_pred = [x.numpy() for x in classifier_pred]

        results = self.bbox_util.detection_out_classifier(classifier_pred, proposal_box, self.config, self.confidence)

        if len(results[0])==0:
            return 
            
        results = np.array(results[0])
        boxes = results[:, :4]
        top_conf = results[:, 4]
        top_label_indices = results[:, 5]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            left, top, right, bottom = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

frcnn = mAP_FRCNN()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # image.save("./input/images-optional/"+image_id+".jpg")
    frcnn.detect_image(image_id,image)
    

print("Conversion completed!")
