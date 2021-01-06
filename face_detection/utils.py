import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def define_fake_4_channels_graph(input_size=(128, 128, 4)):
    inputs = tf.keras.layers.Input(input_size)
    x = tf.keras.layers.Lambda(lambda x: x[:,:,:,:3], name='slicing_inputs')(inputs) # slicing into 3-channels
    model = tf.keras.models.Model(inputs, x)
    sess = K.get_session()
    fake_graph = sess.graph_def
    K.clear_session()
    return fake_graph

def get_pretrained_tflite_weights(model_path):
    model = tf.lite.Interpreter(model_path)
    tensor_details = model.get_tensor_details()
    weights_dict = {}
    layer_names = []
    
    for idx in range(0, len(tensor_details)):
        try:
            name = tensor_details[idx]['name']
            weights = model.get_tensor(idx)
            weights_dict[name] = weights
        except:
            name = tensor_details[idx]['name']
            layer_names.append(name)
    return weights_dict, layer_names

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        for idx, n in enumerate(node.input):
            print(u'└─── %d ─ %s' % (idx, n))

def set_pretrained_weights(model, weights_dict, layer_names):
    for name in layer_names:
        if name.find('conv') != -1:
            pretrained_weights = []
            kernel_weight = weights_dict.get(name+'/Kernel')
            bias_weight = weights_dict.get(name+'/Bias')
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
            pretrained_weights.append(kernel_weight)
            pretrained_weights.append(bias_weight)
            layer = model.get_layer(name)
            layer.set_weights(pretrained_weights)
            
        elif name.find('classificator_') != -1:
            pretrained_weights = []
            kernel_weight = weights_dict.get(name+'/Kernel')
            bias_weight = weights_dict.get(name+'/Bias')
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
            pretrained_weights.append(kernel_weight)
            pretrained_weights.append(bias_weight)
            layer = model.get_layer(name)
            layer.set_weights(pretrained_weights)
            
        elif name.find('regressor_') != -1:
            pretrained_weights = []
            kernel_weight = weights_dict.get(name+'/Kernel')
            bias_weight = weights_dict.get(name+'/Bias')
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
            pretrained_weights.append(kernel_weight)
            pretrained_weights.append(bias_weight)
            layer = model.get_layer(name)
            layer.set_weights(pretrained_weights)
    print("[INFO] Set all pretrained weights")

def convert_to_pb(model, out_path):
    sess = K.get_session()
    output_names = [node.op.name for node in model.outputs]
    frozen_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
    with tf.gfile.GFile(out_path, 'w') as f:
        f.write(frozen_def.SerializeToString())
    print("[INFO] Save frozen graph model in %s"%(out_path))
