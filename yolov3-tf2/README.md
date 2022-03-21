This is the instruction of using my code to buiuld your own yolo-v3 network.

Platform: Python 3.7 + numpy 1.19.5 + tensorflow-gpu 2.4.1

Tranining Part
1. You need to put your training images into 'VOCdevkit/VOC2007/JPEGImages'
2. You need to put your training xmls into 'VOCdevkit/VOC2007/Annotations', noticing that names of images should be equal to names of corresponding xmls
3. You need to create your classes names .txt file and set its path in line27 of  'train.py', the existed 'coco_classes.txt' is the example
4. Set path of your weights saved .h5 file in line79 of 'train.py' 
5. Run 'train.py'
6. After Training, you can find weights saved .h5 file in your setting path.

Options:
1. You can change the dividing of training dataset and validation dataset in line59 of 'preprocessing.py'
2. You can change the batch size in line39 of 'train.py'
3. You can change the epoch in line70 of 'train.py'

Testing Part
1. You need to create your classes names .txt file and set its path in line139 of 'predict.py'
2. Set path of your weights saved .h5 file in line147 of 'predict.py' 
2. Run 'predict.py'
3. The processed image is in 'results' folder
4. you can change the processed image saved path and name in line129 of 'predict.py'

Option:
1. You can match iou(line132 of 'predict.py') and score(line133 of 'predict.py') randomly to obtain the best perfoemance of the network