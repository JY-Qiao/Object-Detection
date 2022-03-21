This is the instruction of using my code to buiuld your own yolo-v4 network.

Platform: Python 3.7 + numpy 1.19.5 + tensorflow-gpu 2.4.1

Tranining Part
1. You need to put your training images into 'VOCdevkit/VOC2007/JPEGImages'
2. You need to put your training xmls into 'VOCdevkit/VOC2007/Annotations', noticing that names of images should be equal to names of corresponding xmls
3. You need to create 'voc_train.txt' file by running 'get_annotations.py'
4. You need to create your classes names .txt file and set its path in line23 of  'train.py', the existed 'voc_classes.txt' is the example
5. You need to create your anchors .txt file and set its path in line25 of  'train.py', the existed 'yolo_anchors.txt' is the example
6. Set path of your weights saved .h5 file in line124 of 'train.py' 
7. Run 'train.py'
8. After Training, you can find weights saved .h5 file in your setting path.

Options:
1. You can change the dividing of training dataset and validation dataset in line62 of 'preprocessing.py'
2. You can change the batch size in line49 of 'train.py'
3. You can change the epoch in line70 of 'train.py'

Testing Part
1. You need to create your anchors .txt file and set its path in line152 of 'predict.py'
1. You need to create your classes names .txt file and set its path in line154 of 'predict.py'
2. Set path of your weights saved .h5 file in line162 of 'predict.py' 
2. Run 'predict.py'
3. The processed image is in 'results' folder
4. you can change the processed image saved path and name in line146 of 'predict.py'

Option:
1. You can match iou(line149 of 'predict.py') and score(line150 of 'predict.py') randomly to obtain the best perfoemance of the network