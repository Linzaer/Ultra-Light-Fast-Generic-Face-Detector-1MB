# Ultra-Light-Fast-Generic-Face-Detector-1MB 
# 超轻量级通用人脸检测模型
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/27.jpg)
该模型设计是针对**边缘计算设备**或**低算力设备**(如用ARM推理)设计的一款实时超轻量级通用人脸检测模型，旨在能在低算力设备中如用ARM进行实时的通用场景的人脸检测推理，当然常规的PC环境（x86 cpu & GPU 同样适用）。有如下几个特点：

 - 在模型大小方面，默认FP32精度下（.pth）文件大小为 **1.1MB**，推理框架int8量化后大小为 **300KB** 左右。
 - 在模型计算量方面，320x240的输入分辨率下仅 **90~109 MFlops**左右，足够轻量。
 - 模型设计有两个版本，version-slim(主干精简速度略快)，version-RFB(加入了修改后的RFB模块，精度更高)。
 - 提供了320x240、640x480不同输入分辨率下使用widerface训练的预训练模型，更好的工作于不同的应用场景。
 - 无特殊算子，支持onnx导出，便于移植推理。


## 测试过正常的运行环境
- Ubuntu16.04、Ubuntu18.04
- Python3.6
- Pytorch1.2
- CUDA10.0 + CUDNN7.6

## 精度、速度、场景测试、模型大小比较
训练集是使用[Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md )提供的清理过的widerface标签配合widerface数据集生成VOC训练集（PS:以下测试结果均为本人测试，结果可能有部分出入）。
### Widerface测试
 - 在WIDER FACE test集测试精度（单尺度输入分辨率：**320*240**） 

模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v2|0.4 |0.04       |0.02
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim|0.765     |0.662       |0.385
version-RFB|**0.784**     |**0.688**       |**0.418**


- 在WIDER FACE test集测试精度（单尺度输入分辨率：**VGA 640*480**） 

模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1|0.197    |0.199       |0.112
libfacedetection v2|0.2 |0.218       |0.147
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |**0.879**|0.807|0.481
version-slim|0.769     |0.733       |0.486
version-RFB|0.851     |**0.81**       |**0.541**

### 终端设备推理速度

- 树莓派4B MNN推理测试耗时 **(单位：ms)**（ARM/A72x4/1.5GHz/输入分辨率 : **320x240** /int8量化） 

模型|1核|2核|3核|4核
------|--------|----------|--------|--------
libfacedetection v1|**28**    |**16**|**12**|9.7
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |46|25|18.5|15
version-slim|29     |**16**       |**12**|**9.5**
version-RFB|35     |19.6       |14.8| 11

### 场景测试
- 若干不同场景视频大致有效人脸检出数量测试（单位：个）（分辨率：**VGA 640*480**，阈值0.6） ：

模型|地铁站室外.MP4(1分43秒)|地铁站内.MP4(1分13秒)|地铁站内2.MP4(36秒)|万达广场室外.MP4(1分36秒)|办公室内.MP4(1分07秒)
------|--------|----------|--------|--------|--------
libfacedetection v1| 6599   |4571|1899|7490|2818
官方 Retinaface-Mobilenet-0.25 (Mxnet) |4415|4897|2026|7882|2557
version-RFB|**10339** |**10444** |**4017**|**13363**|**3403**

### 模型大小比较
- 若干主流开源轻量级人脸检测模型大小比较 ：

模型|模型文件大小（MB）
------|--------
libfacedetection v1（caffe）| 2.58
libfacedetection v2（caffe）| 3.34
官方 Retinaface-Mobilenet-0.25 (Mxnet) | 1.68
version-slim| **1.04**
version-RFB| **1.11** 

## 生成VOC格式训练数据集以及训练流程

1. 下载widerface官网数据集或者下载我提供的训练集解压放入./data文件夹内：

  （1）过滤掉10px*10px 小人脸后的干净widerface数据压缩包 ：[百度云盘 提取码：x5gt](https://pan.baidu.com/s/1m600pp-AsNot6XgIiqDlOw )
  
  （2）未过滤小人脸的完整widerface数据压缩包 ：[百度云盘 提取码：8748](https://pan.baidu.com/s/1ijvZFSb3l7C63Nbz7i6IuQ )
  
2. **（PS:如果下载的是过滤后的上述(1)中的数据包，则不需要执行这步）** 由于widerface存在很多极小的不清楚的人脸，不利于高效模型的收敛，所以需要过滤，默认过滤人脸大小10像素*10像素以下的人脸。
运行./data/wider_face_2_voc_add_landmark.py
```Python
 python3 ./data/wider_face_2_voc_add_landmark.py
```
程序运行和完毕后会在./data目录下生成 **wider_face_add_lm_10_10**文件夹，该文件夹数据和数据包（1）解压后相同，完整目录结构如下：
```Shell
  data/
    retinaface_labels/
      test/
      train/
      val/
    wider_face/
      WIDER_test/
      WIDER_train/
      WIDER_val/
    wider_face_add_lm_10_10/
      Annotations/
      ImageSets/
      JPEGImages/
    wider_face_2_voc_add_landmark.py
```

3. 至此VOC训练集准备完毕，项目根目录下分别有 **train_mb_tiny_fd.sh** 和 **train_mb_tiny_RFB_fd.sh** 两个脚本，前者用于训练**slim版本**模型，后者用于训练**RFB版本**模型，默认参数已设置好，参数如需微调请参考 **./train.py** 中关于各训练超参数的说明。

4. 运行**train_mb_tiny_fd.sh**和**train_mb_tiny_RFB_fd.sh**即可
```Shell
sh train_mb_tiny_fd.sh 或者 sh train_mb_tiny_RFB_fd.sh
```

## 检测图片效果（输入分辨率：640x480）
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/26.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/2.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/4.jpg)
## PS

 - 若生产实际场景为中近距离、人脸大、人脸数少，则建议采用输入尺寸input_size：320（320x240）分辨率训练，并采用320x240图片大小输入进行预测推理，如使用提供的预训练模型**Mb_Tiny_RFB_FD_train_input_320.pth**进行推理。
 - 若生产实际场景为中远距离、人脸中小、人脸数多，则建议采用：
 
 （1）最优：输入尺寸input_size：640（640x480）分辨率训练，并采用640x480图片大小或者更大输入尺寸进行预测推理,如使用提供的预训练模型**Mb_Tiny_RFB_FD_train_input_640.pth**进行推理，更高的召回，更低的误报。
 
 （2）次优：输入尺寸input_size：320（320x240）分辨率训练，并采用640x480图片大小输入进行预测推理，这样对于小人脸更敏感，但是误报会增加。
 
 - 各个场景的最佳效果需要调整输入分辨率从而在速度和精度中间取得平衡。
 - 过大的输入分辨率虽然会增强小人脸的召回率，但是也会提高大、近距离人脸的误报率，而且推理速度延迟成倍增加。
 - 过小的输入分辨率虽然会明显加快推理速度，但是会大幅降低小人脸的召回率。
 - 生产场景的输入分辨率尽量与模型训练时的输入分辨率保持一致，上下浮动不宜过大。

## TODO LIST

 - 加入widerface测试代码
 - 完善部分测试数据
 - 添加MNN、NCNN C++推理代码
 
##  Reference
 - [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
 - [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
 - [RFBNet](https://github.com/ruinmessi/RFBNet)
 - [RFSong-779](https://github.com/songwsx/RFSong-779)
 - [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)

 
##  如果有帮助，欢迎star!
