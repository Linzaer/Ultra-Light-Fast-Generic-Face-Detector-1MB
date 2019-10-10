# Ultra-Light-Fast-Generic-Face-Detector-1MB 
# 超轻量级通用人脸检测模型
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/5.jpg)
该模型设计是针对**边缘计算设备**或**低算力设备**(如用ARM推理)设计的一款实时超轻量级通用人脸检测模型，旨在能在低算力设备中用ARM进行实时的通用场景的人脸检测推理，当然常规的PC环境（x86 cpu & GPU 同样适用）。有如下几个特点：

 - 在模型大小方面，默认FP32精度下（.pth）文件大小为 **1.1MB**，推理框架int8量化后代大小约为 **300KB** 左右。
 - 在模型计算量方面，320x240的输入分辨率下仅 **90~109 MFlops**左右，足够轻量。
 - 模型设计有两个版本，version-slim(主干精简速度略快)，version-RFB(加入了修改后的RFB模块，精度更高)。
 - 提供了320x240、640x480不同输入分辨率下使用wideface训练的预训练模型，更好的工作于不同的应用场景。


## 测试过正常的运行环境
- Ubuntu16.04、Ubuntu18.04
- Python3.6
- Pytorch1.2
- CUDA10.0 + CUDNN7.6

## 精度、速度、场景测试
训练集是使用[Retiaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md )提供的清理过的wideface标签配合widerface数据集生成VOC训练集（PS:以下测试结果均为本人测试，结果可能有部分出入）。
### Widerface测试
 - 在WIDER FACE test集测试精度（单尺度输入分辨率：**VGA 320*240**） 

模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v2|0.4 |0.04       |0.02
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim|0.765     |0.662       |0.385
version-RFB|0.784     |0.688       |0.418


- 在WIDER FACE test集测试精度（单尺度输入分辨率：**VGA 640*480**） 

模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1|0.197    |0.199       |0.112
libfacedetection v2|0.2 |0.218       |0.147
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |0.879|0.807|0.481
version-slim|0.769     |0.733       |0.486
version-RFB|0.851     |0.81       |0.541

### 终端设备推理速度

- 树莓派4B MNN推理测试耗时 **(ms)**（ARM/A72x4/1.5GHz/输入分辨率 : **320x240** /int8量化） 

模型|1核|2核|3核|4核
------|--------|----------|--------|--------
libfacedetection v1|28    |16|12|9.7
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |46|25|18.5|15
version-slim|29     |16       |12|9.5
version-RFB|TODO     |TODO       |TODO|TODO

### 场景测试
- 若干不同场景视频大致有效人脸检出数量测试：
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/scene_test.png)

## 生成VOC格式训练数据集以及训练流程

1. 下载widerface官网数据集或者下载我提供的训练集解压放入./data文件夹内：

  （1）过滤掉10px*10px 小人脸后的干净widerface数据压缩包 ：[百度云盘 提取码：x5gt](https://pan.baidu.com/s/1m600pp-AsNot6XgIiqDlOw )
  
  （2）未过滤小人脸的完整widerface数据压缩包 ：[百度云盘 提取码：8748](https://pan.baidu.com/s/1ijvZFSb3l7C63Nbz7i6IuQ )
  
2. **（PS:如果下载的是过滤后的数据包（1），则不需要执行这步）** 由于widerface存在很多极小的不清楚的人脸，不利于高效模型的收敛，所以需要过滤,默认过滤 人脸大小10像素*10像素以下的人脸。运行./data/wider_face_2_voc_add_landmark.py
```Python
 python3 ./data/wider_face_2_voc_add_landmark.py
```
程序运行和完毕后会在./data目录下生成 **wider_face_add_lm_10_10**文件夹，该文件夹数据和数据包（1）解压后相同。

3. 至此VOC训练集准备完毕，项目根目录下分别有**train_mb_tiny_fd.sh**和**train_mb_tiny_RFB_fd.sh**两个脚本，前者用于训练slim版本模型，后者用于训练RFB版本模型，默认参数已设置好，参数如需微调请参考./train.py中关于各训练超参数的说明。

4. 运行**train_mb_tiny_fd.sh**和**train_mb_tiny_RFB_fd.sh**即可
```Shell
sh train_mb_tiny_fd.sh 或者 sh train_mb_tiny_RFB_fd.sh
```

## 检测图片效果
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/1.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/2.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/4.jpg)
## PS

 - 若生产实际场景为中近距离、人脸大、人脸数少，则建议采用输入尺寸input_size：320（320x240）分辨率训练，并采用320x240图片大小输入进行预测推理。
 - 若生产实际场景为中远距离、人脸中小、人脸数多，则建议采用：
 
 （1）输入尺寸input_size：320（320x240）分辨率训练，并采用640x480图片大小输入进行预测推理。
 
 （2）输入尺寸input_size：640（640x480）分辨率训练，并采用640x480图片大小或者更大输入尺寸输入进行预测推理。
 
 - 各个场景的最佳效果需要调整输入分辨率从而在速度和精度中间取得平衡。
 - 过大的输入分辨率虽然会增强小人脸的召回率，但是也会提高大、近距离人脸的误报率，而且推理速度延迟成倍增加。
 - 过小的输入分辨率虽然会明显加快推理速度，但是会大幅降低小人脸的召回率。
 - 生产场景模型输入分辨率尽量与模型巡礼看输入分辨率保持一致。

## TODO LIST

 - 加入widerface测试代码
 - 完善部分测试数据
 
##  Reference
 - [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
 - [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
 - [RFBNet](https://github.com/ruinmessi/RFBNet)
 - [Retiaface]https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)
