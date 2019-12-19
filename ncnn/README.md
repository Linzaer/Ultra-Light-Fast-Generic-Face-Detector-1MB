# C++ implemententation of [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) with [NCNN](https://github.com/Tencent/ncnn)

## Build

```bash
git clone --recursive --depth=1 https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

cd Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn

mkdir build && cd build && cmake ..
make -j$(nproc)
```

## Run

```bash
./main ../data/version-RFB/RFB-320.bin ../data/version-RFB/RFB-320.param ../data/test.jpg
```
* We provide converted NCNN models of version-slim-320 and version-RFB-320 in ./ncnn/data .

## How to convert pretrained model to ncnn

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
* But the exported onnx model may contains many redundant operators such as Shape, Gather and Unsqueeze that is not supported in ncnn.

```
Shape not supported yet!
Gather not supported yet!
  # axis=0
Unsqueeze not supported yet!
  # axes 7
Unsqueeze not supported yet!
  # axes 7
```

Fortunately, we can use this tool to eliminate them :
https://github.com/daquexian/onnx-simplifier

```
python3 -m onnxsim  version-RFB-320_without_postprocessing.onnx version-RFB-320_simplified.onnx

```

Next, you can convert this onnx model like **version-RFB-320_simplified.onnx** into a ncnn model. Here is a website for online conversion : https://convertmodel.com/?tdsourcetag=s_pctim_aiomsg. You can also use the NCNN compiled conversion tool **onnx2ncnn**.

## PS
* If you want to run faster, try using the version-slim model or using lower-resolution inputs like 160x120 or 128x96.

## Result
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/ncnn/data/result.jpg)