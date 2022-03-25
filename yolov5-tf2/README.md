This is the instruction of using my code to buiuld your own yolo-v5 network.

Platform: Python 3.7 + numpy 1.19.5 + tensorflow-gpu 2.4.1

Tranining Part
1. You need to put your training images into 'VOCdevkit/VOC2007/JPEGImages'

2. You need to put your training xmls into 'VOCdevkit/VOC2007/Annotations', noticing that names of images should be equal to names of corresponding xmls

3. You need to create 'voc_train.txt' file by running 'get_annotations.py'

4. You need to create your classes names .txt file and set its path in line23 of  'train.py', the existed 'voc_classes.txt' is the example

   ```python
   cls_txt_path = os.path.join('model_data', 'yolo_classes.txt')
   ```

5. You need to create your anchors .txt file and set its path in line25 of  'train.py', the existed 'yolo_anchors.txt' is the example

   ```python
   anchors_path = os.path.join('model_data', 'yolo_anchors.txt')
   ```

6. Set path of your weights saved .h5 file in line132 of 'train.py' 

   ```python
   model.save_weights('logs/yolo5_x_weights.h5')
   ```

7. Run 'train.py'

8. After Training, you can find weights saved .h5 file in your setting path.

Options:
1. You can change the dividing of training dataset and validation dataset in line67 of 'train.py'

   ```python
   train_count = int(len(annotations) * 0.8)
   val_count = len(annotations) - train_count
   ```

2. You can change the batch size in line48 of 'train.py'

   ```python
   batch_size = 2
   ```

3. You can change the size of your network in line51 of 'train.py'

   ```python
   size = 'x'
   ```

4. You can change the epoch in line75 of 'train.py'

   ```python
   EPOCHS = 100  # 总训练世代
   ```

Testing Part
1. You need to create your anchors .txt file and set its path in line144 of 'predict.py'

   ```python
   anchors = load_anchors('model_data/yolo_anchors.txt')
   ```

2. You need to create your classes names .txt file and set its path in line146 of 'predict.py'

   ```python
   classes_name = load_classes('model_data/yolo_classes.txt')
   ```

3. Set path of your weights saved .h5 file in line153 of 'predict.py' 

   ```python
   model.load_weights('logs/yolov5_x.h5')
   ```

4. Set the size of your trained network in line147 of 'predict.py'

   ```python
   size = 'x'
   ```

5. Run 'predict.py'

6. The processed image is in 'results' folder

7. you can change the processed image saved path and name in line137 of 'predict.py'

   ```python
   cv2.imwrite('results/pred.jpg', img)
   ```

Option:
1. You can match iou(line140 of 'predict.py') and score(line141 of 'predict.py') randomly to obtain the best perfoemance of the network

   ```python
   iou = 0.5  # 设置极大值抑制阈值
   score = 0.3  # 设置置信度阈值
   ```

   