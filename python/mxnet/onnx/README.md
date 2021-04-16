<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->

# ONNX Export Support for MXNet

### Overview
[ONNX](https://onnx.ai/), or Open Neural Network Exchange, is an open source deep learning model format that acts as a framework neutral graph representation between DL frameworks or between training and inference. With the ability to export models to the ONNX format, MXNet users can enjoy faster inference and a wider range of deployment device choices, including edge and mobile devices where MXNet installation may be constrained. Popular hardware-accelerated and/or cross-platform ONNX runtime frameworks include Nvidia [TensorRT](https://github.com/onnx/onnx-tensorrt), Microsoft [ONNXRuntime](https://github.com/microsoft/onnxruntime), Apple [CoreML](https://github.com/onnx/onnx-coreml) and [TVM](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html), etc. 

### ONNX Versions Supported
ONNX 1.7 -- Fully Supported
ONNX 1.8 -- Work in Progress

### Installation
From the 1.9 release and on, the ONNX export module has become an offical, built-in module in MXNet. You can access the module at `mxnet.onnx`. 

If you are a user of earlier MXNet versions and do not want to upgrade MXNet, you can still enjoy the latest ONNX suppor by pulling the MXNet source code and building the wheel for only the mx2onnx module. Just do `cd python/mxnet/onnx` and then build the wheel with `python3 -m build`. You should be able to find the wheel under `python/mxnet/onnx/dist/mx2onnx-0.0.0-py3-none-any.whl` and install it with `pip install mx2onnx-0.0.0-py3-none-any.whl`. You should be able to access the module with `import mx2onnx` then.

### APIs

### Operator Support Matrix
|MXNet Op|ONNX Version|
|:-|:-:|
|Activation|1.7 1.8 |
|BatchNorm|1.7 1.8 |
|BlockGrad|1.7 1.8 |
|Cast|1.7 1.8 |
|Concat|1.7 1.8 |
|Convolution|1.7 1.8 |
|Crop|1.7 1.8 |
|Deconvolution|1.7 1.8 |
|Dropout|1.7 1.8 |
|Embedding|1.7 1.8 |
|Flatten|1.7 1.8 |
|FullyConnected|1.7 1.8 |
|InstanceNorm|1.7 1.8 |
|L2Normalization|1.7 1.8 |
|LRN|1.7 1.8 |
|LayerNorm|1.7 1.8 |
|LeakyReLU|1.7 1.8 |
|LogisticRegressionOutput|1.7 1.8 |
|MakeLoss|1.7 1.8 |
|Pad|1.7 1.8 |
|Pooling|1.7 1.8 |
|RNN|1.7 1.8 |
|ROIPooling|1.7 1.8 |
|Reshape|1.7 1.8 |
|SequenceMask|1.7 1.8 |
|SequenceReverse|1.7 1.8 |
|SliceChannel|1.7 1.8 |
|SoftmaxOutput|1.7 1.8 |
|SwapAxis|1.7 1.8 |
|UpSampling|1.7 1.8 |
|_arange|1.7 1.8 |
|_contrib_AdaptiveAvgPooling2D|1.7 1.8 |
|_contrib_BilinearResize2D|1.7 1.8 |
|_contrib_ROIAlign|1.7 1.8 |
|_contrib_arange_like|1.7 1.8 |
|_contrib_box_decode|1.7 1.8 |
|_contrib_box_nms|1.7 1.8 |
|_contrib_div_sqrt_dim|1.7 1.8 |
|_contrib_interleaved_matmul_selfatt_qk|1.7 1.8 |
|_contrib_interleaved_matmul_selfatt_valatt|1.7 1.8 |
|_copy|1.7 1.8 |
|_div_scalar|1.7 1.8 |
|_equal_scalar|1.7 1.8 |
|_greater_scalar|1.7 1.8 |
|_lesser_scalar|1.7 1.8 |
|_linalg_gemm2|1.7 1.8 |
|_maximum|1.7 1.8 |
|_maximum_scalar|1.7 1.8 |
|_minimum|1.7 1.8 |
|_minimum_scalar|1.7 1.8 |
|_minus_scalar|1.7 1.8 |
|_mul_scalar|1.7 1.8 |
|_ones|1.7 1.8 |
|_plus_scalar|1.7 1.8 |
|_power|1.7 1.8 |
|_power_scalar|1.7 1.8 |
|_random_normal|1.7 1.8 |
|_random_uniform|1.7 1.8 |
|_random_uniform_like|1.7 1.8 |
|_rdiv_scalar|1.7 1.8 |
|_rminus_scalar|1.7 1.8 |
|_rnn_param_concat|1.7 1.8 |
|_sample_multinomial|1.7 1.8 |
|_zeros|1.7 1.8 |
|abs|1.7 1.8 |
|add_n|1.7 1.8 |
|arccos|1.7 1.8 |
|arcsin|1.7 1.8 |
|arctan|1.7 1.8 |
|argmax|1.7 1.8 |
|argmin|1.7 1.8 |
|argsort|1.7 1.8 |
|batch_dot|1.7 1.8 |
|broadcast_add|1.7 1.8 |
|broadcast_axis|1.7 1.8 |
|broadcast_div|1.7 1.8 |
|broadcast_equal|1.7 1.8 |
|broadcast_greater|1.7 1.8 |
|broadcast_greater_equal|1.7 1.8 |
|broadcast_lesser|1.7 1.8 |
|broadcast_lesser_equal|1.7 1.8 |
|broadcast_like|1.7 1.8 |
|broadcast_logical_and|1.7 1.8 |
|broadcast_logical_or|1.7 1.8 |
|broadcast_logical_xor|1.7 1.8 |
|broadcast_minimum|1.7 1.8 |
|broadcast_mod|1.7 1.8 |
|broadcast_mul|1.7 1.8 |
|broadcast_power|1.7 1.8 |
|broadcast_sub|1.7 1.8 |
|broadcast_to|1.7 1.8 |
|ceil|1.7 1.8 |
|clip|1.7 1.8 |
|cos|1.7 1.8 |
|depth_to_space|1.7 1.8 |
|dot|1.7 1.8 |
|elemwise_add|1.7 1.8 |
|elemwise_div|1.7 1.8 |
|elemwise_mul|1.7 1.8 |
|elemwise_sub|1.7 1.8 |
|exp|1.7 1.8 |
|expand_dims|1.7 1.8 |
|floor|1.7 1.8 |
|gather_nd|1.7 1.8 |
|hard_sigmoid|1.7 1.8 |
|identity|1.7 1.8 |
|log|1.7 1.8 |
|log2|1.7 1.8 |
|log_softmax|1.7 1.8 |
|logical_not|1.7 1.8 |
|max|1.7 1.8 |
|mean|1.7 1.8 |
|min|1.7 1.8 |
|negative|1.7 1.8 |
|norm|1.7 1.8 |
|null|1.7 1.8 |
|one_hot|1.7 1.8 |
|ones_like|1.7 1.8 |
|prod|1.7 1.8 |
|reciprocal|1.7 1.8 |
|relu|1.7 1.8 |
|repeat|1.7 1.8 |
|reshape_like|1.7 1.8 |
|reverse|1.7 1.8 |
|shape_array|1.7 1.8 |
|sigmoid|1.7 1.8 |
|sin|1.7 1.8 |
|size_array|1.7 1.8 |
|slice|1.7 1.8 |
|slice_axis|1.7 1.8 |
|slice_like|1.7 1.8 |
|softmax|1.7 1.8 |
|space_to_depth|1.7 1.8 |
|sqrt|1.7 1.8 |
|square|1.7 1.8 |
|squeeze|1.7 1.8 |
|stack|1.7 1.8 |
|sum|1.7 1.8 |
|take|1.7 1.8 |
|tan|1.7 1.8 |
|tanh|1.7 1.8 |
|tile|1.7 1.8 |
|topk|1.7 1.8 |
|transpose|1.7 1.8 |
|where|1.7 1.8 |
|zeros_like|1.7 1.8 |

### GluonCV Pretrained Model Support Matrix

### GluonNLP Pretrained Model Support Matrix