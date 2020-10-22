# Tensorflow implementation of Ultra-Light-Fast-Generic-Face-Detector-1MB with converter

You can use this script to converter origin model to tensorflow version.

## Run
Covert model
```Python
 python3 ./convert_tensorflow.py --net_type <RFB|slim>
```

Inference on image
```Python
 python3 ./det_image.py --net_type <RFB|slim> --img_path <path>
```

## Result
![img1](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter/blob/master/imgs/test_output_RFB.jpg)

## Reference
- [Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter)
