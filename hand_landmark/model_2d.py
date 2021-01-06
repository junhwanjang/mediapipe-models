# Hand landmark 2d model
from tensorflow.keras.layers import Input, Conv2D, PReLU, SeparableConv2D, Add, ReLU, MaxPooling2D, Reshape, Lambda, Activation, DepthwiseConv2D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_blocks(x, num_filter, activation='prelu', num_iterations=1):
    for num_iter in range(0, num_iterations):
        if activation == 'prelu':
            x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
        elif activation == 'relu':
            x = ReLU()(x)
        
        shortcut = x
        x = Conv2D(int(num_filter / 2), kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([shortcut, x])
    return x

def conv_blocks_with_pooling(x, num_filter, activation='prelu', padding=False, pad_value=None):
    if activation == 'prelu':
        x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    elif activation == 'relu':
        x = ReLU()(x)
    
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(shortcut)

if padding:
    paddings = tf.constant([[0,0],[0,0],[0,0],[0,pad_value]])
    shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut) # CONSTANT / REFLECT / SYMMETRIC
    x = Conv2D(int(num_filter / 2), kernel_size=(2, 2), strides=(2, 2), padding='valid', use_bias=True)(x)
    x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])
    else:
        x = Conv2D(int(num_filter / 2), kernel_size=(2, 2), strides=(2, 2), padding='valid', use_bias=True)(x)
        x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([x, shortcut])
    return x

def hand_landmark_2d_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(inputs)
    
    # block 1 ~ 9 (1, 128, 128, 16)
    x = conv_blocks(x, 16, activation='prelu', num_iterations=8)
    x = conv_blocks_with_pooling(x, 32, activation='prelu', padding=True, pad_value=16)
    
    # block 10 ~ 18 (1, 64, 64, 32)
    x = conv_blocks(x, 32, activation='prelu', num_iterations=8)
    x = conv_blocks_with_pooling(x, 64, activation='prelu', padding=True, pad_value=32)
    
    # block 19 ~ 27 (1, 32, 32, 64)
    x = conv_blocks(x, 64, activation='prelu', num_iterations=8)
    x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(shortcut)
    paddings = tf.constant([[0,0],[0,0],[0,0],[0,192]])
    shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut)
    x = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='valid', use_bias=True)(x)
    x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])
    
    # block 28 ~ 36 (1, 16, 16, 256)
    x = conv_blocks(x, 256, activation='prelu', num_iterations=8)
    x = conv_blocks_with_pooling(x, 256, activation='prelu', padding=False)
    
    # block 37 ~ 45 (1, 8, 8, 256)
    x = conv_blocks(x, 256, activation='prelu', num_iterations=8)
    x = conv_blocks_with_pooling(x, 256, activation='prelu', padding=False)
    
    # block 46 ~ 54 (1, 4, 4, 256)
    x = conv_blocks(x, 256, activation='prelu', num_iterations=8)
    x = conv_blocks_with_pooling(x, 256, activation='prelu', padding=False)
    
    # block 55 ~ 63 (1, 2, 2, 256)
    x = conv_blocks(x, 256, activation='prelu', num_iterations=8)
    
    # Last layer (1, 2, 2, 256)
    x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    ## Hand_flag value (1, 1)
    hand_flag = Conv2D(1, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=True, name='conv_handflag')(x)
    hand_flag = Activation('sigmoid', name='activation_handflag')(hand_flag)
    hand_flag = Reshape(target_shape=(1,), name='output_handflag')(hand_flag)
    ## Hand_Landmark_2d (1, 42)
    landmarks = Conv2D(42, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=True, name='convld_21_2d')(x)
    landmarks = Reshape(target_shape=(42,), name='ld_21_2d')(landmarks)
    
    model = Model(inputs, [landmarks, hand_flag])
    return model
