import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Model

from nets.centernet_training import loss
from nets.hourglass import HourglassNetwork
from nets.resnet import ResNet50, centernet_head


def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=100):

    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    #-------------------------------------------#
    #   Obtain (b, 128 * 128 * 80)
    #-------------------------------------------#
    hm = tf.reshape(hm, (b, -1))
    #-----------------------------#
    #   (b, k), (b, k)
    #-----------------------------#
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)

    #--------------------------------------#
    #   Calculate class,(xs,ys),indices
    #--------------------------------------#
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=100,num_classes=20):
    #-----------------------------------------------------#
    #   hm          b, 128, 128, num_classes 
    #   wh          b, 128, 128, 2 
    #   reg         b, 128, 128, 2 
    #   scores      b, max_objects
    #   indices     b, max_objects
    #   class_ids   b, max_objects
    #   xs          b, max_objects
    #   ys          b, max_objects
    #-----------------------------------------------------#
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    
    #-----------------------------------------------------#
    #   wh          b, 128 * 128, 2
    #   reg         b, 128 * 128, 2
    #-----------------------------------------------------#
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    #-----------------------------------------------------#
    #   batch_idx   b, max_objects
    #-----------------------------------------------------#
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) + tf.reshape(indices, [-1])
                    
    #-----------------------------------------------------#
    #   Get the corresponding parameters of the top_k anchor boxes
    #-----------------------------------------------------#
    topk_reg = tf.gather(tf.reshape(reg, [-1,2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    
    topk_wh = tf.gather(tf.reshape(wh, [-1,2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    #-----------------------------------------------------#
    #   Use parameters to get the center of the adjusted prediction box
    #   topk_cx     b,k,1
    #   topk_cy     b,k,1
    #-----------------------------------------------------#
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    #-----------------------------------------------------#
    #   Calculate the upper left and lower right corners of the prediction box
    #   topk_x1     b,k,1       The x-axis coordinate of the upper left corner of the prediction box
    #   topk_y1     b,k,1       The y-axis coordinate of the upper left corner of the prediction box
    #   topk_x2     b,k,1       The x-axis coordinate of the lower right corner of the prediction box
    #   topk_y2     b,k,1       The y-axis coordinate of the lower right corner of the prediction box
    #-----------------------------------------------------#
    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    
    #-----------------------------------------------------#
    #   scores      b,k,1       Score of prediction box
    #   class_ids   b,k,1       Type of prediction box
    #-----------------------------------------------------#
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

    #-----------------------------------------------------#
    #   detections  Stacking of all parameters of the prediction box
    #   The first four parameters are the coordinates of the prediction box, the last two are the score and type of the prediction box
    #-----------------------------------------------------#
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'hourglass']
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone=='resnet50':
        #-----------------------------------#
        #   512, 512, 3 -> 16, 16, 2048
        #-----------------------------------#
        C5 = ResNet50(image_input)
        #--------------------------------------------------------------------------------------------------------#
        #   16, 16, 2048 -> 32, 32, 256 -> 64, 64, 128 -> 128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                                                              -> 128, 128, 64 -> 128, 128, 2
        #                                                              -> 128, 128, 64 -> 128, 128, 2
        #--------------------------------------------------------------------------------------------------------#
        y1, y2, y3 = centernet_head(C5,num_classes)

        if mode=="train":
            loss_ = Lambda(loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
            return model
        else:
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                                num_classes=num_classes))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
            return prediction_model

    else:
        outs = HourglassNetwork(image_input,num_stacks,num_classes)

        if mode=="train":
            loss_all = []
            for out in outs:  
                y1, y2, y3 = out
                loss_ = Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
                loss_all.append(loss_)
            loss_all = Lambda(tf.reduce_mean,name='centernet_loss')(loss_all)

            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=loss_all)
            return model
        else:
            y1, y2, y3 = outs[-1]
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                                num_classes=num_classes))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=[detections])
            return prediction_model
