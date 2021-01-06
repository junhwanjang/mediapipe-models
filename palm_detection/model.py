from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, ReLU, MaxPooling2D, Reshape, Lambda, Activation, DepthwiseConv2D, Conv2D, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def conv_blocks(x, num_filter, num_iterations=1):
    for num_iter in range(0, num_iterations):
        x = ReLU()(x)
        shortcut = x
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([shortcut, x])
    return x

def conv_blocks_with_pooling(x, num_filter, padding=False, pad_value=None):
    x = ReLU()(x)
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(shortcut)

    if padding:
        paddings = tf.constant([[0,0],[0,0],[0,0],[0,pad_value]])
        shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut) # CONSTANT / REFLECT / SYMMETRIC
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([x, shortcut])
    else:
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([x, shortcut])
    return x

def palm_detection_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(inputs)

    # block 1 ~ 8 (1, 128, 128, 32)
    x = conv_blocks(x, 32, num_iterations=7)
    x = conv_blocks_with_pooling(x, 64, padding=True, pad_value=32)

    # block 9 ~ 16 (1, 64, 64, 64)
    x = conv_blocks(x, 64, num_iterations=7)
    x = conv_blocks_with_pooling(x, 128, padding=True, pad_value=64)

    # block 17 ~ 24 (1, 32, 32, 128)
    x = conv_blocks(x, 128, num_iterations=7)
    x = ReLU()(x)
    shortcut_1 = x #(1, 32, 32, 128)
    
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(shortcut)
    paddings = tf.constant([[0,0],[0,0],[0,0],[0,128]])
    shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut) # CONSTANT / REFLECT / SYMMETRIC
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])

    # block 25 ~ 32 (1, 16, 16, 256)
    x = conv_blocks(x, 256, num_iterations=7)
    x = ReLU()(x)
    shortcut_2 = x #(1, 16, 16, 256)

    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(shortcut)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])

    # block 33 ~ 39 (1, 8, 8, 256)
    x = conv_blocks(x, 256, num_iterations=7)
    
    # Last layers
    ## (1, 8, 8, 256)
    x = ReLU()(x)
    shortcut_3 = x # (1, 8, 8, 256)
    x = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same', use_bias=True)(x)
    x = ReLU()(x)
    x = Add()([x, shortcut_2])
    
    ## (1, 16, 16, 256)
    shortcut = x
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])
    
    x = ReLU()(x)
    shortcut_4 = x # (1, 16, 16, 256)
    x = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same', use_bias=True)(x)
    x = ReLU()(x)
    x = Add()([x, shortcut_1])

    ## (1, 32, 32, 128)
    shortcut = x
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)

    ### Last block - Classificators
    classificator_8 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='classificator_8')(x)
    classificator_16 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='classificator_16')(shortcut_4)
    classificator_32 = Conv2D(6, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='classificator_32')(shortcut_3)
    #### Reshape and Concatenation
    classificator_8 = Reshape(target_shape=(-1, 1))(classificator_8) # (1, 2048, 1)
    classificator_16 = Reshape(target_shape=(-1, 1))(classificator_16) # (1, 512, 1)
    classificator_32 = Reshape(target_shape=(-1, 1))(classificator_32) # (1, 384, 1)
    classificator_concat = Concatenate(axis=1, name='classificators')([classificator_8, classificator_16, classificator_32]) # (1, 2944, 1)
    
    ### Last block - Regressors
    regressor_8 = Conv2D(36, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='regressor_8')(x)
    regressor_16 = Conv2D(36, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='regressor_16')(shortcut_4)
    regressor_32 = Conv2D(108, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True, name='regressor_32')(shortcut_3)
    #### Reshape and Concatenation
    regressor_8 = Reshape(target_shape=(-1, 18))(regressor_8) # (1, 2048, 18)
    regressor_16 = Reshape(target_shape=(-1, 18))(regressor_16) # (1, 512, 18)
    regressor_32 = Reshape(target_shape=(-1, 18))(regressor_32) # (1, 384, 18)
    regressor_concat = Concatenate(axis=1, name='regressors')([regressor_8, regressor_16, regressor_32])

    model = Model(inputs, [regressor_concat, classificator_concat])
    return model
