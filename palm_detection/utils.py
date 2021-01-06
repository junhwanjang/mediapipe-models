import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def define_fake_4_channels_graph(input_size=(256, 256, 4)):
    inputs = tf.keras.layers.Input(input_size)
    x = tf.keras.layers.Lambda(lambda x: x[:,:,:,:3], name='slicing_inputs')(inputs) # slicing into 3-channels
    model = tf.keras.models.Model(inputs, x)
    sess = K.get_session()
    fake_graph = sess.graph_def
    K.clear_session()
    return fake_graph

def set_pretrained_weights(model, weights_dir='pretrained_weights/'):
    # Get layer names
    layer_names = [
    'classificator_8', 'classificator_16', 'classificator_32', 
    'regressor_8', 'regressor_16', 'regressor_32',
    'conv2d', 'depthwise_conv2d', 'conv2d_transpose', 'conv2d_transpose_1'
    ]
    num_conv = 41
    num_depth_conv = 40
    # Append conv2d layer names
    for i in range(1, num_conv+1):
        name = 'conv2d_' + str(i)
        layer_names.append(name)
    for i in range(1, num_depth_conv+1):
        name = 'depthwise_conv2d_' + str(i)
        layer_names.append(name)
    layer_names.sort()

    # Set pretrained weights from npy file
    for name in layer_names:
        pretrained_weights = []
    kernel_weight_path = weights_dir + name + "_Kernel.npy"
    bias_weight_path = weights_dir + name + "_Bias.npy"
    kernel_weight = np.load(kernel_weight_path)
    bias_weight = np.load(bias_weight_path)
    if name.find("conv2d_transpose") == -1:
        kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
    else:
        kernel_weight = kernel_weight.transpose(1, 2, 0, 3)
    
    pretrained_weights.append(kernel_weight)
    pretrained_weights.append(bias_weight)
    layer = model.get_layer(name)
    layer.set_weights(pretrained_weights)
    
    print("[INFO] Set all pretrained weights")

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        for idx, n in enumerate(node.input):
            print(u'└─── %d ─ %s' % (idx, n))
