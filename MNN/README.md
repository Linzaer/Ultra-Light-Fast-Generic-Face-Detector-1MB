# C++ implemententation of [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) with [MNN](https://github.com/alibaba/MNN)

## Build

```bash
git clone --recursive --depth=1 https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

cd Ultra-Light-Fast-Generic-Face-Detector-1MB/MNN
```

* Replace  **libMNN.so** under ./mnn/lib with your compiled libMNN.so and then :

```bash
mkdir build && cd build && cmake ..
make -j$(nproc)
```

## Run
* Use FP32 model and run in FP16 mode:
```bash
./Ultra-face-mnn ../model/version-RFB/RFB-320.mnn  ../imgs/1.jpg ../imgs/2.jpg ../imgs/3.jpg ../imgs/4.jpg
```
* Use quantized INT8 model:
```bash
./Ultra-face-mnn ../model/version-RFB/RFB-320-quant-KL-5792.mnn  ../imgs/1.jpg ../imgs/2.jpg ../imgs/3.jpg ../imgs/4.jpg
```

* We provide both converted MNN FP32 and **quantized INT8** models of version-slim-320 and version-RFB-320 in ./MNN/model . The xxx-quant-KL-xxx.mnn is quantified by the **KL** method and xxx-quant-ADMM-xxx.mnn is quantified by the **ADMM** method.

## How to convert pretrained model to MNN

* Code bellow (```vision/ssd/ssd.py```) should be commented out when convert pytorch pretrained model to onnx. Comment it out and use the **convert_to_onnx.py** in official repo to finish this step.

```python
if self.is_test:
    confidences = F.softmax(confidences, dim=2)
    boxes = locations # this line should be added.
    #boxes = box_utils.convert_locations_to_boxes(
    #    locations, self.priors, self.config.center_variance, self.config.size_variance
    #)
    # boxes = box_utils.center_form_to_corner_form(boxes) # these lines should be commented out. detail information and analyze comming soon.
    return confidences, boxes
else:
    return confidences, locations
```
Then you can generate the onnx model like **version-RFB-320_without_postprocessing.onnx** in onnx directory. (You need to rename your model when convert.)
* Then we can use this tool to simplify onnx :
https://github.com/daquexian/onnx-simplifier

```
python3 -m onnxsim  version-RFB-320_without_postprocessing.onnx version-RFB-320_simplified.onnx

```

Next, you can convert this onnx model like **version-RFB-320_simplified.onnx** into a MNN model. Here is a website for online conversion : https://convertmodel.com. You can also use the MNN compiled conversion tool **MNNConvert**.



## PS
* Since MNN mainly accelerates  model inference on mobile, so the INT8 quantified model will run slower on **PC** than FP32 model in CPU mode.
* If you want to run faster, try using the version-slim model ,using lower-resolution inputs like 160x120 /128x96 or using quantified models(On the mobile).

## Result
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/MNN/result.jpg)
