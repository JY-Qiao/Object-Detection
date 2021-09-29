import colorsys
import os
import time

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont

from nets.centernet import centernet
from utils.utils import centernet_correct_boxes, letterbox_image, nms


def preprocess_image(image):
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383 , 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


class CenterNet(object):
    _defaults = {
        "model_path"        : 'model_data/centernet_resnet50_voc.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        "backbone"          : 'resnet50',
        # "model_path"        : 'model_data/centernet_hourglass_coco.h5',
        # "classes_path"      : 'model_data/coco_classes.txt',
        # "backbone"          : 'hourglass',
        "input_shape"       : [512,512,3],
        "confidence"        : 0.3,

        "nms"               : True,
        "nms_threhold"      : 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        

        self.num_classes = len(self.class_names)


        self.centernet = centernet(self.input_shape,num_classes=self.num_classes,backbone=self.backbone,mode='predict')
        self.centernet.load_weights(self.model_path, by_name=True)
        
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    @tf.function
    def get_pred(self, photo):
        preds = self.centernet(photo, training=False)
        return preds

    def detect_image(self, image):
        image = image.convert('RGB')
        
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])

        photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]

        photo = np.reshape(preprocess_image(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        
        preds = self.get_pred(photo).numpy()

        if self.nms:
            preds = np.array(nms(preds,self.nms_threhold))

        if len(preds[0])<=0:
            return image

        preds[0][:,0:4] = preds[0][:,0:4]/(self.input_shape[0]/4)
        
        det_label = preds[0][:, -1]
        det_conf = preds[0][:, -2]
        det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])

        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])

        photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]

        photo = np.reshape(preprocess_image(photo), [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        preds = self.get_pred(photo).numpy()
        
        if self.nms:
            preds = np.array(nms(preds, self.nms_threhold))

        if len(preds[0])>0:
            preds[0][:, 0:4] = preds[0][:, 0:4] / (self.input_shape[0] / 4)
            
            det_label = preds[0][:, -1]
            det_conf = preds[0][:, -2]
            det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
            
            boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.get_pred(photo).numpy()
            
            if self.nms:
                preds = np.array(nms(preds, self.nms_threhold))

            if len(preds[0])>0:
                preds[0][:, 0:4] = preds[0][:, 0:4] / (self.input_shape[0] / 4)
                
                det_label = preds[0][:, -1]
                det_conf = preds[0][:, -2]
                det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
                
                boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
