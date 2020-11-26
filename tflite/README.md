# TFLite implementation of Ultra-Light-Fast-Generic-Face-Detector-1MB

TFLite model is suitable for edge computing devices.
Please refer to the official [Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) for learning how to deploy the face detection model on your phones.

![Dwq3dS.jpg](https://s3.ax1x.com/2020/11/26/Dwq3dS.jpg)

## Run

Inference on image

``` bash
python3 inference_test.py --net_type <RFB|slim> --img_path <IMG_PATH>
```

Inference on video

``` bash
python3 inference_test.py --net_type <RFB|slim> --video_path <VIDEO_PATH>
```

## Import

``` python
from TFLiteFaceDetector import UltraLightFaceDetecion

fd = UltraLightFaceDetecion(model_path,
                            input_size=(320, 240), conf_threshold=0.6,
                            center_variance=0.1, size_variance=0.2,
                            nms_max_output_size=200, nms_iou_threshold=0.3)
```

## Files Tree

The pretrained weights are converted form `onnx -> protobuf -> tflite`.

``` bash
.
├── model  # keras defined model architecture
│   ├── tflite_RFB_320_without_postprocessing.py
│   └── tflite_slim_320_without_postprocessing.py
├── pretrained  # pretrained model without post-processing
│   ├── version-RFB-320_without_postprocessing.tflite
│   └── version-slim-320_without_postprocessing.tflite
├── README.md
├── inference_test.py  # detector test script
└── TFLiteFaceDetector.py  # class file of the tflite detector
```

## Special Thanks

Part of the code for this work is referenced from the following repositories:

- [Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter](https://github.com/jason9075/Ultra-Light-Fast-Generic-Face-Detector_Tensorflow-Model-Converter)
