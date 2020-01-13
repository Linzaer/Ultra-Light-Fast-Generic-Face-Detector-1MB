# Python implemententation of [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) with [MNN](https://github.com/alibaba/MNN)

## How to use MNN in Python

### Install
#### Install Depencies
##### graphviz
for macOS:
```bash
brew install graphviz
```
for Linux:
```bash
apt-get install graphviz
```

#### Python Version Limitation
Python2.7, 3.5, 3.6, 3.7 are supported, but for Windows, python2.7 is not supported.
for macOS:
```bash
pip install -U MNN
```

for Linux:
As PyPi requires all wheels to be tagged with "ManyLinux", and old version pip can't get the "ManyLinux" Tagged wheel, thus you have to upgrade your pip to newer version in order to use "pip install"
```bash
pip install -U pip
pip install -U MNN
```

## Run
* Use FP32 model(version-RFB) and run in FP16 mode:
```bash
python ultraface_py_mnn.py  --model_path ../model/version-RFB/RFB-320.mnn
```
* Use quantized INT8 model:
```bash
python ultraface_py_mnn.py  --model_path ../model/version-RFB/RFB-320-quant-KL-5792.mnn 
```

* We provide both converted MNN FP32 and **quantized INT8** models of version-slim-320 and version-RFB-320 in ./MNN/model . The xxx-quant-KL-xxx.mnn is quantified by the **KL** method and xxx-quant-ADMM-xxx.mnn is quantified by the **ADMM** method.


## PS
* Since MNN mainly accelerates  model inference on mobile, so the INT8 quantified model will run slower on **PC** than FP32 model in CPU mode.
* If you want to run faster, try using the version-slim model ,using lower-resolution inputs like 160x120 /128x96 or using quantified models(On the mobile).

## Result
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/MNN/result.jpg)