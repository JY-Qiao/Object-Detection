import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input

from yolo import yolo_body

from PIL import Image, ImageDraw, ImageFont
import cv2


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# 从保存好类别的txt文件中读取数据集的类别
def load_classes(file_path):
    with open(file_path) as f:
        classes_name = f.readlines()
        classes = [class_name.strip() for class_name in classes_name]
    return classes

# 从保存好先验框尺寸的txt文件中读取先验框的尺寸
def load_anchors(file_path):
    with open(file_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape((3, 3, 2))  # .shape = [3,3,2]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_y_pres(y_pres, input_size, anchors):
    # .shape = [1,13,13,255] and [1,26,26,255] and [1,52,52,255]
    ratio = input_size / int(y_pres.shape[1])
    y_pres = np.reshape(y_pres, [y_pres.shape[-3], y_pres.shape[-2], 3, y_pres.shape[-1] // 3])  # resize为[13,13,3,85]，以13为例
    # 获得网格点坐标
    cor_x = np.reshape(np.tile(np.arange(y_pres.shape[0]), [y_pres.shape[1]]), (y_pres.shape[0], y_pres.shape[1], 1, 1))
    cor_y = np.transpose(cor_x, (1,0,2,3))
    cor_xy = np.tile(np.concatenate([cor_x,cor_y],-1), [1, 1,y_pres.shape[2], 1])
    # 计算中心点坐标
    pre_xy = (sigmoid(y_pres[...,:2]) + cor_xy) * ratio
    # 计算预测框宽高
    anchor_grid = np.reshape(np.array(anchors),(1,1,3,2))
    pre_wh = np.exp(y_pres[...,2:4]) * anchor_grid
    # 计算置信度
    pre_con = sigmoid(y_pres[...,4:5])
    # 计算每一类的概率
    pre_cls = sigmoid(y_pres[...,5:])
    # 计算中心点的坐标
    pre_wh_half = pre_wh / 2
    pre_mins = pre_xy - pre_wh_half
    pre_maxes = pre_xy + pre_wh_half

    return np.concatenate([pre_mins, pre_maxes, pre_con, pre_cls], axis=-1)

def get_IOU(box1, box2):
    box1, box2 = np.array(box1), np.array(box2)
    # 计算交叠面积
    intersect_min = np.maximum(box1[:2], box2[:2])
    intersect_max = np.minimum(box1[2:4], box2[2:4])
    intersect_wh = np.maximum(intersect_max-intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # 计算box1的面积
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])

    # 计算box2的面积
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])

    # 计算IOU
    union_area = box1_area + box2_area - intersect_area
    IOU = intersect_area / union_area

    return IOU

def soft_nms(boxes, iou, sigma=0.5):
    # 按照con值对boxes进行从大到小的排列
    boxes = sorted(boxes.tolist(), key=(lambda x:x[4]), reverse=True)
    nms_boxes = []
    while len(boxes) > 0:
        nms_boxes.append(boxes[0])
        if len(boxes) == 1:
            break
        del boxes[0]
        # 计算boxes[0]与剩下的boxes的IOU
        IOU_list = []
        for box in boxes:
            IOU_list.append(get_IOU(nms_boxes[-1][:4], box[:4]))
            IOU_array = np.array(IOU_list)
            box[4] = np.exp(-(np.dot(IOU_array.T, IOU_array)) / sigma) * box[4]
        # 重新排序
        boxes = sorted(boxes, key=(lambda x: x[4]), reverse=True)
        j = 0
        for _ in range(len(boxes)):
            IOU = get_IOU(nms_boxes[-1][:4], boxes[j][:4])
            if IOU > iou:
                del boxes[j]
                j -= 1
            j += 1
    return nms_boxes

def draw_result(boxes, img, class_names):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        xmin = box[0] * img.size[0] / 416
        ymin = box[1] * img.size[1] / 416
        xmax = box[2] * img.size[0] / 416
        ymax = box[3] * img.size[1] / 416

        bias = np.ones_like(xmin) * 5

        xmin = xmin - bias
        ymin = ymin + bias
        xmax = xmax + bias
        ymax = ymax - bias

        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymin = min(img.size[1], np.floor(ymin + 0.5).astype('int32'))
        xmax = min(img.size[0], np.floor(xmax + 0.5).astype('int32'))
        ymax = max(0, np.floor(ymax + 0.5).astype('int32'))

        # 画预测框
        label = str(class_names[int(box[6])]) + ' ' + str(format(box[5] * 100, '.2f')) + '%'
        font = ImageFont.truetype("setting_files/simhei.ttf", size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness = max((img.size[0] + img.size[1]) // 400, 1)
        label_size = draw.textsize(label, font)
        label = label.encode('UTF-8')

        if ymax - label_size[1] >= 0:
            text_origin = np.array([xmin, ymax - label_size[1]])
        else:
            text_origin = np.array([xmin, ymax + 1])

        for i in range(thickness):
            draw.rectangle([xmin + i, ymin - i, xmax - i, ymax + i], fill=None, outline=(255, 255, 0))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 0))

        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite('results/pred.jpg', img)

def main():
    iou = 0.5  # 设置极大值抑制阈值
    score = 0.3  # 设置置信度阈值
    # 获得先验框的尺寸
    anchors = load_anchors('model_data/yolo_anchors.txt')
    # 获得voc数据集的所有类别
    classes_name = load_classes('model_data/voc_classes.txt')

    # 加载模型
    model_input = Input(shape=(None,None,3))
    model = yolo_body(model_input,3,len(classes_name))

    print('已加载模型')
    # 加载权重文件
    model.load_weights('logs/yolo4_weight.h5')
    print('已加载权重')

    while True:
        # 打开需要预测的图片，并将其转换成float32类型
        img = input('Input image filepath:')
        try:
            img = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            image = img.resize(size=(416,416))
            image = np.array(image, dtype='float32')
            image /= 255.
            image = np.expand_dims(image, axis=0)  # 将图片数据转换为[1,416,416,3]的形式
            # 获得预测结果
            y_pres = model.predict(image)

            boxes = []
            idx = 0
            for y_pre in y_pres:
                # y_pre的形状为[1,13,13,85*3] or [1,26,26,85*3] or [1,52,52,85*3]]
                new_y_pre = get_y_pres(y_pre,416,anchors[idx])
                # new_y_pre的形状为[13,13,3,85] or [26,26,3,85] or [52,52,3,85]
                for i in range(int(y_pre.shape[2])):
                    for j in range(int(y_pre.shape[1])):
                        for k in range(3):
                            con = new_y_pre[i,j,k,4]
                            classes_prob = new_y_pre[i,j,k,5:]
                            classes = con * classes_prob
                            classes = classes.tolist()
                            index = classes.index(max(classes))
                            # 获得预测框最有可能包含的物体的概率和种类
                            new_y_pre[i,j,k,5:7] = (max(classes), index)

                            # 挑选置信度大于score的预测框
                            if con > score:
                                boxes.append(new_y_pre[i,j,k,:7])

                idx += 1  # 循环预测下一个yolo head

            # 极大化抑制
            boxes = soft_nms(np.array(boxes), iou)
            # 绘制最终结果
            draw_result(boxes, img, classes_name)

if __name__ == '__main__':
    main()