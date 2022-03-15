This is the instruction of using my code to buiuld your own yolo-v3 network.

Tranining Part
1. You need to put your training images into 'VOCdevkit/VOC2007/JPEGImages'
2. You need to put your training xmls into 'VOCdevkit/VOC2007/Annotations', noticing that names of images should be equal to names of corresponding xmls
3. You need to create your classes names .txt file and set its path in line27 of  'train.py', the existed 'coco_classes.txt' is the example
4. Set path of your weights saved .h5 file in line78 of 'train.py' 
5. Run 'train.py'
6. After Training, you can find weights save .h5 file in your setting path.
Options:
1. You can change the dividing of training dataset and validation dataset in line123 of 'preprocessing.py'
2. You can change the buffer size in line57 of 'train.py'
3. You can change the batch size in line39 of 'train.py'
4. You can change the epoch in line69 of 'train.py'

Testing Part
1. You need to create your classes names .txt file and set its path in line133 of 'predict.py'
2. Run 'predict.py'
3. The processed image is in results folder
Option:
1. You can match iou(line128 of 'predict.py') and score(line129 of 'predict.py') randomly to obtain the best perfoemance of the network

Tips:
If you want to use my network directly without training, you need to do the following steps:
1. You need to set path in line133 of 'predict.py' as ''logs/yolo_weights.h5 
1. You need to change value of idx in line154 of 'predict.py' into 2.
2. You need to change code in line174 of 'predict.py' into idx -= 1
3. Run 'predict.py'