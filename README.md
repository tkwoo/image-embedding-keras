# Image embedding - keras
triplet loss based image embedding implementation using keras

## Requirements

- Python 3.6
- OpenCV 3.4.0 (option: build from src with highgui)
- [Keras 2.2.2](https://github.com/fchollet/keras)
- [TensorFlow 1.12.0](https://github.com/tensorflow/tensorflow)

## Usage  

First, check directory structure

    ├── train.py
    ├── network.py
    ├── utils.py
    ├── preprocessor.py
    ├── checkpoint
        ├── base_model_pre_weights.h5
        └── weights.xx.h5
    └── result
        └── save the generated images when training


To train a model

    $ python train.py


-----------


### Reference

To be added