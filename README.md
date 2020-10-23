# EndImage
基于Flask + 部分自训练的模型实现的后人脸以及人体姿态相关的后端，附上Postman接口测试JSON
文件

对接的前端：https://github.com/shwanliu/FrontEndImage


### app.py 后端入口
### 1. ------ faceAttr (人脸属性) (ResNet50 + CelebA) 
来自：https://github.com/shwanliu/FANet
#### ------------ checkpoints（模型目录）
#### ------------  models （网络结构）
### 2. ------ faceEmotioon （人脸情绪）(ResNet50 + Fer2013) 数据集后续更换ExW 
#### ------------ checkpoints（模型目录）
#### ------------  models （网络结构）

### 3. ------ faceSwap （人脸交换）
#### ------------ checkpoints（模型目录）
#### ------------  models （网络结构）

### 4. ------ lioghtOpenPose （人体姿态）
#### ------------ checkpoints（模型目录）
#### ------------  models （网络结构）
来自：https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
