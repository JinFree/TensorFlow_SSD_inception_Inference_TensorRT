#!/bin/bash
echo "TensorRT optimization of TF trained SSD inception in Jetson"
echo "TensorFlow must be installed!"
echo "install pycuda"
pip3 install pycuda
echo "prepare tensorRT samples"
cp -r /usr/src/tensorrt/samples ~/trt_samples
echo "build engine"
cd ~/trt_samples/python/uff_ssd
python3 detect_objects.py images/image1.jpg
