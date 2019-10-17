# Ultra-Light-Fast-Generic-Face-Detector-1MB 
# Ultra-lightweight face detection model
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/27.jpg)
The model design is a real-time ultra-lightweight universal face detection model designed for **edge computing devices** or **low computing devices** (such as ARM reasoning), which can be used in low-power computing devices such as ARM The face detection reasoning of the real-time general scene is also applicable to the mobile terminal and the PC.

  - In terms of model size, the default FP32 precision (.pth) file size is **1.04~1.1MB**, and the inference frame int8 is quantized to a size of **300KB**.
  - In the model calculation, the input resolution of 320x240 is about **90~109 MFlops**.
  - There are two versions of the model, version-slim (slightly faster simplification), version-RFB (with the modified RFB module, higher precision).
  - Provides a pre-training model using the widerface training at 320x240 and 640x480 different input resolutions to better work in different application scenarios.
  - Support onnx export for easy porting reasoning.

## Tested the normal operating environment
- Ubuntu16.04、Ubuntu18.04、Windows 10（inference）
- Python3.6
- Pytorch1.2
- CUDA10.0 + CUDNN7.6

## Accuracy, speed, model size comparison
The training set is a cleaned widerface tag provided with [Retinaface] (https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md) with the widerface dataset to generate a VOC training set (PS: the following test) The results are tested by myself and the results may be partially inconsistent).
### Widerface test
  - Test accuracy in the WIDER FACE test set (single-scale input resolution: **320*240 or scaling by the maximum side length of 320)**

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim|0.765     |0.662       |0.385
version-RFB|**0.784**     |**0.688**       |**0.418**


- Test accuracy in the WIDER FACE test set (single-scale input resolution: ** VGA 640 * 480 or scaling by the maximum side length of 640)

Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.741 |0.683       |0.421
libfacedetection v2（caffe）|0.773 |0.718       |0.485
Retinaface-Mobilenet-0.25 (Mxnet)   |**0.879**|0.807|0.481
version-slim|0.757     |0.721       |0.511
version-RFB|0.851     |**0.81**       |**0.541**

> - This part is mainly to test the effect of the test set at medium and small resolution.
> - RetinaFace-mnet (Retinaface-Mobilenet-0.25), from a great job [insightface] (https://github.com/deepinsight/insightface), when testing the network, the original image is 320 by the maximum side length or 640 is scaled, so the face will not be deformed, and the rest of the network will have a fixed size resize. At the same time, the result of the RetinaFace-mnet optimal 1600 single-scale test test set was 0.887 (Easy) / 0.87 (Medium) / 0.791 (Hard).

### Terminal device reasoning speed

- Raspberry Pi 4B MNN reasoning test time ** (unit: ms)** (ARM/A72x4/1.5GHz/input resolution: **320x240** /int8 quantization)

Model|1 core|2 core|3 core|4 core
------|--------|----------|--------|--------
libfacedetection v1|**28**    |**16**|**12**|9.7
Official Retinaface-Mobilenet-0.25 (Mxnet)   |46|25|18.5|15
version-slim|29     |**16**       |**12**|**9.5**
version-RFB|35     |19.6       |14.8| 11


### Model size comparison
- Comparison of several open source lightweight face detection models:

Model|model file size（MB）
------|--------
libfacedetection v1（caffe）| 2.58
libfacedetection v2（caffe）| 3.34
Official Retinaface-Mobilenet-0.25 (Mxnet) | 1.68
version-slim| **1.04**
version-RFB| **1.11** 

## Generate VOC format training data set and training process

1. Download the wideface official website dataset or download the training set I provided and extract it into the ./data folder:

   (1) Filter out the clean widerface data compression package after 10px*10px face: [Baiyun cloud disk extraction code: x5gt] (https://pan.baidu.com/s/1m600pp-AsNot6XgIiqDlOw)
  
   (2) Complete wideface data compression package for unfiltered small faces: [Baiyun cloud disk extraction code: xeis] (https://pan.baidu.com/s/1Qusz-CjIzsILmjv6jtFpXQ)
  
2. ** (PS: If you download the filtered packets in (1) above, you don't need to perform this step)** Because the wideface has many small and unclear faces, it is not conducive to efficient models. Convergence, so you need to filter the training, the default is to filter the face size of 10 pixels x 10 pixels.
run./data/wider_face_2_voc_add_landmark.py
```Python
 python3 ./data/wider_face_2_voc_add_landmark.py
```
After the program is run and finished, the **wider_face_add_lm_10_10** folder will be generated in the ./data directory. The folder data and data package (1) are the same after decompression. The complete directory structure is as follows:
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

3. At this point, the VOC training set is ready. There are two scripts: **train_mb_tiny_fd.sh** and **train_mb_tiny_RFB_fd.sh** in the root directory of the project. The former is used to train the **slim version** model, and the latter is used. Training **RFB version** model, the default parameters have been set, if the parameters need to be fine-tuned, please refer to the description of each training parameter in **./train.py**.

4. Run **train_mb_tiny_fd.sh** and **train_mb_tiny_RFB_fd.sh**
```Shell
sh train_mb_tiny_fd.sh 或者 sh train_mb_tiny_RFB_fd.sh
```

## Detecting image effects (input resolution: 640x480)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/26.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/2.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/4.jpg)
## PS

 - If the actual production scene is medium-distance, large face, and small number of faces, it is recommended to use input size input_size: 320 (320x240) resolution training, and use 320x240 image size input for predictive reasoning, such as using the provided pre-training The model **Mb_Tiny_RFB_FD_train_input_320.pth** is reasoned.
 - If the actual production scene is medium to long distance, small face, and large number of faces, it is recommended to:
 
 (1) Optimal: input size input_size: 640 (640x480) resolution training, and use the same or larger input size for predictive reasoning, such as using the provided pre-training model **Mb_Tiny_RFB_FD_train_input_640.pth** for reasoning, lower False positives.
 
 (2) Sub-optimal: input size input_size: 320 (320x240) resolution training, and use 480x360 or 640x480 size input for predictive reasoning, more sensitive to small faces, false positives will increase.
 
 - The best results for each scene require adjustment of the input resolution to strike a balance between speed and accuracy.
 - Excessive input resolution will enhance the recall rate of small faces, but it will also increase the false positive rate of large and close-range faces, and the speed of reasoning will increase exponentially.
 - Too small input resolution will significantly speed up the reasoning, but it will greatly reduce the recall rate of small faces.
 - The input resolution of the production scene should be as consistent as possible with the input resolution of the model training, and the up and down floating should not be too large.

## TODO LIST

  - Join the widerface test code
  - Improve some test data
  - Add MNN, NCNN C++ inference code
 
##  Reference
 - [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
 - [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
 - [RFBNet](https://github.com/ruinmessi/RFBNet)
 - [RFSong-779](https://github.com/songwsx/RFSong-779)
 - [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)
