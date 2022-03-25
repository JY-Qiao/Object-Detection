This is the instruction of using my code to buiuld your own yolo-v3 network.

Platform: Python 3.7 + numpy 1.19.5 + tensorflow-gpu 2.4.1

Tranining Part
1. You need to put your training images into 'VOCdevkit/VOC2007/JPEGImages'

2. You need to put your training xmls into 'VOCdevkit/VOC2007/Annotations', noticing that names of images should be equal to names of corresponding xmls

3. You need to create your classes names .txt file and set its path in line28 of  'train.py', the existed 'coco_classes.txt' is the example

   ```python
   cls_path = 'coco_classes.txt'
   ```

4. Set path of your weights saved .h5 file in line84 of 'train.py' 

   ```python
   model.save_weights('logs/yolo_v3_weights.h5')
   ```

5. Run 'train.py'

6. After Training, you can find weights saved .h5 file in your setting path.

Options:
1. You can change the dividing of training dataset and validation dataset in line65 of 'train.py'

   ```python
   train_count = int(len(imgs_path) * 0.8)
   test_count = len(imgs_path) - train_count
   ```

2. You can change the batch size in line40 of 'train.py'

   ```python
   batch_size = 2
   ```

3. You can change the epoch in line75 of 'train.py'

   ```python
   EPOCHS = 100
   ```

Testing Part
1. You need to create your classes names .txt file and set its path in line138 of 'predict.py'

   ```python
   classes_name = load_coco_classes('coco_classes.txt')
   ```

2. Set path of your weights saved .h5 file in line146 of 'predict.py' 

   ```python
   yolo_model.load_weights('logs/yolo_weights.h5')
   ```

3. Run 'predict.py'

4. The processed image is in 'results' folder

5. you can change the processed image saved path and name in line128 of 'predict.py'

   ```python
   cv2.imwrite('results/pred.jpg', img)
   ```

Option:
1. You can match iou(line131 of 'predict.py') and score(line132 of 'predict.py') randomly to obtain the best perfoemance of the network

```python
    iou = 0.5  # 设置极大值抑制阈值
    score = 0.3  # 设置置信度阈值
```

