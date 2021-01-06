from tensorflow.keras.layers import Input, Conv2D, PReLU, DepthwiseConv2D, Add, ReLU, MaxPooling2D, Reshape, Lambda, Activation
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
import tensorflow as tf

# Mediapipe: hand_landmark_3d.tflite
def conv_blocks(x, num_filter, activation='relu', num_iterations=1):
    for num_iter in range(0, num_iterations):
        if activation == 'relu':
            x = ReLU()(x)
        elif activation == 'prelu':
            x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
        shortcut = x
        x = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([shortcut, x])
    return x

def conv_blocks_with_pooling(x, num_filter, activation='prelu', padding=False, pad_value=None):
    if activation == 'relu':
        x = ReLU()(x)
    elif activation == 'prelu':
        x = PReLU(alpha_initializer='uniform', shared_axes=[1, 2])(x)
    
    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(shortcut)

    if padding:
        paddings = tf.constant([[0,0],[0,0],[0,0],[0,pad_value]])
        shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut) # CONSTANT / REFLECT / SYMMETRIC
        x = DepthwiseConv2D(kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([shortcut, x])
    else:
        x = DepthwiseConv2D(kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=True)(x)
        x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        x = Add()([shortcut, x])
    return x

def hand_landmark_3d_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)(inputs)

    # block 1 ~ 5 (1, 128, 128, 32)
    x = conv_blocks(x, 32, activation='prelu', num_iterations=1)
    x = conv_blocks(x, 32, activation='relu', num_iterations=3)
    x = conv_blocks_with_pooling(x, 64, activation='relu', padding=True, pad_value=32)

    # block 6 ~ 11 (1, 64, 64, 64)
    x = conv_blocks(x, 64, activation='relu', num_iterations=5)
    x = conv_blocks_with_pooling(x, 128, activation='relu', padding=True, pad_value=64)

    # block 12 ~ 18 (1, 32, 32, 128)
    x = conv_blocks(x, 128, activation='relu', num_iterations=6)
    x = conv_blocks_with_pooling(x, 192, activation='relu', padding=True, pad_value=64)

    # block 19 ~ 25 (1, 16, 16, 192)
    x = conv_blocks(x, 192, activation='relu', num_iterations=6)
    x = conv_blocks_with_pooling(x, 192, activation='relu', padding=False)

    # block 26 ~ 32 (1, 8, 8, 192)
    x = conv_blocks(x, 192, activation='relu', num_iterations=6)
    x = conv_blocks_with_pooling(x, 192, activation='relu', padding=False)

    # block 33 ~ 39 (1, 4, 4, 192)
    x = conv_blocks(x, 192, activation='relu', num_iterations=6)
    x = conv_blocks_with_pooling(x, 192, activation='relu', padding=False)

    # block 40 ~ 45 (1, 2, 2, 192)
    x = conv_blocks(x, 192, activation='relu', num_iterations=6)

    # Last layer
    x = ReLU()(x)
    ## output_handflag
    hand_flag = Conv2D(1, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=True, name='conv_handflag')(x)
    hand_flag = Activation('sigmoid', name='activation_handflag')(hand_flag)
    hand_flag = Reshape(target_shape=(1,), name='output_handflag')(hand_flag)

    ## Hand_Landmark_3d (1, 63)
    landmarks = Conv2D(63, kernel_size=(2, 2), strides=(1, 1), padding='valid', use_bias=True, name='convld_21_3d')(x)
    landmarks = Reshape(target_shape=(63,), name='ld_21_3d')(landmarks)

    model = Model(inputs, [hand_flag, landmarks])
    return model
