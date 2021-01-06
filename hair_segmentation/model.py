# Real-time Hair Segmentation and Recoloring on Mobile GPUs (https://arxiv.org/abs/1907.06740)
import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, PReLU, MaxPooling2D, Add, Concatenate
from keras.layers import Input, Conv2D, Conv2DTranspose, PReLU, MaxPooling2D, Add, Concatenate
## MaxUnpooling2D --> Input: tensor and Argmaxed Tensor --> How to?
## MaxPoolingWithArgmax2D --> Compatible with tf.nn.max_pool_with_argmax
## 
## TODO
