import math

import numpy as np
import tensorflow as tf
from PIL import Image


class BBoxUtility(object):
    def __init__(self, overlap_threshold=0.7, ignore_threshold=0.3, rpn_pre_boxes=6000, rpn_nms=0.7, classifier_nms=0.3, top_k=300):
        self.overlap_threshold  = overlap_threshold
        self.ignore_threshold   = ignore_threshold
        self.rpn_pre_boxes      = rpn_pre_boxes

        self.rpn_nms            = rpn_nms
        self.classifier_nms     = classifier_nms
        self.top_k              = top_k

    def iou(self, box):

        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        area_true = (box[2] - box[0]) * (box[3] - box[1])

        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])

        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_ignore_box(self, box, return_iou=True):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))

        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        assigned_priors = self.priors[assign_mask]

        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        self.num_priors = len(anchors)
        self.priors = anchors

        assignment = np.zeros((self.num_priors, 4 + 1))

        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment

        apply_along_axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1

        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)

        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]

        assignment[:, 4][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        # Get the width and height of the prior box
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # Get the center point of the prior box
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # The xy-axis offset of the true box from the center of the prior box
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y

        # Get the width and height of the real box
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height

        # Get the upper left and lower right corners of the real box
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # The stacking of upper left and lower right corners of the real box
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out_rpn(self, predictions, mbox_priorbox):
        #---------------------------------------------------#
        #   Get the confidence
        #---------------------------------------------------#
        mbox_conf = predictions[0]
        #---------------------------------------------------#
        #   mbox_loc is the result of regress prediction
        #---------------------------------------------------#
        mbox_loc = predictions[1]
        #---------------------------------------------------#
        #   Get the prior box
        #---------------------------------------------------#
        mbox_priorbox = mbox_priorbox

        results = []

        for i in range(len(mbox_loc)):

            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)

            c_confs = mbox_conf[i, :, 0]
            argsort_index = np.argsort(c_confs)[::-1]
            c_confs = c_confs[argsort_index[:self.rpn_pre_boxes]]
            decode_bbox = decode_bbox[argsort_index[:self.rpn_pre_boxes], :]

            idx = tf.image.non_max_suppression(decode_bbox, c_confs, self.top_k, iou_threshold=self.rpn_nms).numpy()

            good_boxes = decode_bbox[idx]
            confs = c_confs[idx][:, None]

            c_pred = np.concatenate((confs, good_boxes), axis=1)
            argsort = np.argsort(c_pred[:, 0])[::-1]
            c_pred = c_pred[argsort]
            results.append(c_pred)
            
        return np.array(results)

    def detection_out_classifier(self, predictions, proposal_box, config, confidence):

        proposal_conf = predictions[0]

        proposal_loc = predictions[1]

        results = []

        for i in range(len(proposal_conf)):
            proposal_pred = []
            proposal_box[i, :, 2] = proposal_box[i, :, 2] - proposal_box[i, :, 0]
            proposal_box[i, :, 3] = proposal_box[i, :, 3] - proposal_box[i, :, 1]
            for j in range(proposal_conf[i].shape[0]):
                if np.max(proposal_conf[i][j, :-1]) < confidence:
                    continue
                label = np.argmax(proposal_conf[i][j, :-1])
                score = np.max(proposal_conf[i][j, :-1])

                (x, y, w, h) = proposal_box[i, j, :]

                (tx, ty, tw, th) = proposal_loc[i][j, 4*label: 4*(label+1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]

                cx = x + w/2.
                cy = y + h/2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                proposal_pred.append([x1,y1,x2,y2,score,label])

            num_classes = np.shape(proposal_conf)[-1]
            proposal_pred = np.array(proposal_pred)
            good_boxes = []
            if len(proposal_pred)!=0:
                for c in range(num_classes):
                    mask = proposal_pred[:, -1] == c
                    if len(proposal_pred[mask]) > 0:
                        boxes_to_process = proposal_pred[:, :4][mask]
                        confs_to_process = proposal_pred[:, 4][mask]
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self.top_k, iou_threshold=self.classifier_nms).numpy()

                        good_boxes.extend(proposal_pred[mask][idx])
            results.append(good_boxes)

        return results
