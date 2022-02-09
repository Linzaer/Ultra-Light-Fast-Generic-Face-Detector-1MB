# Masked Face Detection 

![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img4.jpeg)

# Extending Ultra-L face model for masked facial detection

Ultra-L face detection model achieves great popularity in edge and client based applications. It has a surprising balance of model size and accuracy performance, e.g.,
- The default FP32 *.pth model size is **1.04~1.1MB**, and the inference framework int8 quantization size is about **300KB**.
- Only **90~109 MFlops** for 320x240 input resolution.
- Supported inference code for [NCNN](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/ncnn), [MNN](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN), [INT8](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN/model),
[Onnx](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/caffe), [OpencvDNN](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/caffe/ultra_face_opencvdnn_inference.py), etc.

COVID-19 has ravaged the world in the past two years, and wearing masks has become the norm in our lives on many occasions. However, most traditional face datasets such as Wider Face currently lack face samples with masks. Therefore, the face detection model based on conventional datasets will fail in the scenario where all attendants wear masks.
[Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) is the most popular face detection model we can find on Github that supports Mask detection. 
However, this model is trained only using 4095 images (2165 masked / 1930 without mask), which is a pretty small dataset. 
We will experience many false positives in the actual application scenarios.

This original intention inspired me to build a larger dataset to provide better open-source masked facial detection models and help the world survive the pandemic.
The main contribution of this project is to provide balanced facial training data combining the [wider_face_add_lm_10_10](https://drive.google.com/open?id=1OBY-Pk5hkcVBX1dRBOeLI4e4OCvqJRnH) and [MAFA face](https://imsg.ac.cn/research/maskedface.html) dataset. The [MAFA](https://imsg.ac.cn/research/maskedface.html) data was converted to pascal-VOC format and merged into the [wider_face_add_lm_10_10](https://drive.google.com/open?id=1OBY-Pk5hkcVBX1dRBOeLI4e4OCvqJRnH).

## About the WIDER_MAFA_Balanced dataset
The *Wider_MAFA_Balanced* dataset (**4.8GB**) can be downloaded at [HPI owncloud](https://owncloud.hpi.de/s/L4MUGqrpeENLbSv).
It contains 38225 images in total where 31084 for training and 7141 for testing, respectively.
The specific composition information is shown in the following table:

Source| Class | Train | Test |Total|
----|------|-------|------|-----
MAFA face| masked_face | 15542 | 3922 | 19464 |
Wider face| face | 12859 | 3219 | 16078 |
*MAFA human body* | face | 2683 | 0 |2683

*MAFA human body* indicates the extracted training samples with human body occlusions.  

I use this script for converting MAFA data format to pascal VOC:
```Shell
masked_face/mafa2voc.py
```

## About the pre-trained models
```Shell
masked_face/
   pretrained/
      RFB-320-masked_face-v2.pth   # trained with 320x240
      RFB-640-masked_face-v2.pth   # trained with 640x480
      RFB-640-masked_face-v2.onnx  # suitable for 640x480
      RFB-1280-masked_face-v2.onnx # suitable for 1280x960
```

## Detection Result (input resolution: 1280x960)

The following visual results are created by using this script:
```Shell
masked_face/detect_imgs.py
```
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img1.jpeg)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img2.jpeg)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img3.jpg)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img5.jpeg)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img6.webp)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img7.webp)
![img1](https://github.com/yanghaojin/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/masked_face/readme_imgs/img8.jpeg)

Author: Haojin Yang
