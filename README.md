# KCPNet

This is the code of our paper "Fine-Grained Recognition for Oriented Ship Against Complex Scenes in Optical Remote in Remote Sensing Images", which is under review. We decide to publish our code in advance.

Note: This is an early version of the code. We will update it in the future.


Introduction
--
Knowledge-Driven Context Perception Network (KCPNet) combines advantages of data-driven approaches and model-driven approaches for infrared ship detection task. 

Since there is no public infrared ship detection dataset, we tested KCPNet on the proposed [ISDD][1], which is a new public dataset for infrared ship detection. Extensive experiments demonstrate that our KCPNet achieves state-of-the-art performance on ISDD.

Experiment Environment
--
Ubuntu / Win10

Python 3.6

Tensorflow 1.10.2

Cuda 10.0

GeForce GTX TITAN


How to use
--

### 1. Download the code and prepare for training.

```
git clone https://github.com/yaqihan-9898/KCPNet  # clone repo
cd KCPNet
pip install -r requirements.txt  # install
```

### 2. Compile

For linux:
```
cd ./data/coco/PythonAPI
python setup.py build_ext --inplace
python setup.py build_ext install
cd ./lib/utils　　
python setup.py build_ext --inplace
```
For windows:
```
cd /d .\data\coco\PythonAPI
python setup.py build_ext --inplace
python setup.py build_ext install
cd /d .\lib\utils　　
python setup.py build_ext --inplace
```
### 3. Prepare data and pretrained model

Put all dataset images under ./data/VOCdevkit2007/VOC2007/JPEGImages

Put all dataset annotations (VOC format) under ./data/VOCdevkit2007/VOC2007/Annotations

Put train.txt, val.txt, test.txt, and trainval.txt under ./data/VOCdevkit2007/VOC2007/ImageSets/Main

Download pretrained weight from [our pretrained model][2] (based on ResNet101) with access code **1596** or [office model][3].

### 4. Train
Modify ./lib/config/config.py:

'network' in Line10

If you want to train your own dataset, please modify:

./lib/config/config.py: 'dataset' in Line20, 'image_ext' in Line21

./lib/datasets/pascal_voc.py: 'CLASSES' in Line24


Go to the root path and start to train:
```
$ python train.py
```


### 5. Demo
```
python demo.py
```

### 6. Eval
```
python eval.py
```


[1]:https://github.com/yaqihan-9898/ISDD
[2]: https://pan.baidu.com/s/1j-WRmj8da2yHsZP1odXfHg
[3]: https://github.com/tensorflow/models/tree/master/research/slim
