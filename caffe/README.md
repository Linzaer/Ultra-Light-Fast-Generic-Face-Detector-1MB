## I added several operator(Transpose/Permute/Softmax) conversion support based on [onnx2caffe](https://github.com/MTlab/onnx2caffe).
# Convert pytorch to Caffe by ONNX
This tool converts [pytorch](https://github.com/pytorch/pytorch) model to [Caffe](https://github.com/BVLC/caffe) model by [ONNX](https://github.com/onnx/onnx)  
only use for inference

### Dependencies
* caffe (with python support)
* pytorch 0.4+ (optional if you only want to convert onnx)
* onnx  

### Current support operation
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten
* Transpose/Permute (new)
* Softmax (new)

## PS
* You need to use [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) to simplify onnx model and then run convertCaffe.py to convert it into caffe model.
* You need to install [ssd-caffe](https://github.com/weiliu89/caffe/tree/ssd) and pycaffe of ssd-caffe.
