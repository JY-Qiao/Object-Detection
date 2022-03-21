# 修改voc_classes.txt后，运行此程序，会在主文件夹内生成voc_train,voc_val,voc_test三个.txt文件

import os
import glob
from os import getcwd
from get_xml import PascalVocXmlParser

set = [('voc', 'train')]
xmls_dir = os.path.join('VOCdevkit', 'VOC2007', 'Annotations', '*.xml')
xmls_path = glob.glob(xmls_dir)

imgs_dir = os.path.join('VOCdevkit', 'VOC2007', 'JPEGImages', '*.jpg')
imgs_path = glob.glob(imgs_dir)

# 读取数据集的类别
def load_classes(file_path):
    with open(file_path) as f:
        class_name = f.readlines()
        classes = [c.strip() for c in class_name]
    return classes

def convert(xml_path, file, classes):
    Parser = PascalVocXmlParser()
    boxes = Parser.get_boxes(xml_path)
    labels = Parser.get_labels(xml_path)
    for i in range(len(boxes)):
        file.write(' ' + ','.join([str(cor) for cor in boxes[i]]) + ',' + str(classes.index(labels[i])))

wd = getcwd()
classes = load_classes('model_data/voc_classes.txt')
for dataset_name, dataset_divided in set:
    file = open('%s_%s.txt'%(dataset_name, dataset_divided), 'w', encoding='utf-8')
    for i in range(len(imgs_path)):
        file.write('%s/%s'%(wd, imgs_path[i]))
        convert(xmls_path[i], file, classes)
        file.write('\n')
    file.close()