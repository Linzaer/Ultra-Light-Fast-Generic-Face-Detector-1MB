import tensorflow as tf

import sys
sys.path.append("../tf")
from backend.op import conv_bn, conv_dw, separable_conv


def create_slim_net(input_shape, base_channel, num_classes):
    input_node = tf.keras.layers.Input(
        shape=(input_shape[0], input_shape[1], 3))

    net = conv_bn(input_node, base_channel, stride=2,
                  prefix='basenet.0')  # 120x160
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.1')
    net = conv_dw(net, base_channel * 2, stride=2, prefix='basenet.2')  # 60x80
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.3')
    net = conv_dw(net, base_channel * 4, stride=2, prefix='basenet.4')  # 30x40
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.5')
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.6')
    header_0 = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.7')
    net = conv_dw(header_0, base_channel * 8, stride=2,
                  prefix='basenet.8')  # 15x20
    net = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.9')
    header_1 = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.10')
    net = conv_dw(header_1, base_channel * 16,
                  stride=2, prefix='basenet.11')  # 8x10
    header_2 = conv_dw(net, base_channel * 16, stride=1, prefix='basenet.12')

    out = tf.keras.layers.Conv2D(
        base_channel * 4, 1, padding='SAME', name='extras_convbias')(header_2)
    out = tf.keras.layers.ReLU(name='extras_relu1')(out)
    out = separable_conv(out, base_channel * 16, kernel_size=3, stride=2, padding=1,
                         prefix='extras_sep')
    header_3 = tf.keras.layers.ReLU(name='extras_relu2')(out)

    reg_0 = separable_conv(header_0, 3 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_0_sep')
    cls_0 = separable_conv(header_0, 3 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_0_sep')

    reg_1 = separable_conv(header_1, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_1_sep')
    cls_1 = separable_conv(header_1, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_1_sep')

    reg_2 = separable_conv(header_2, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_2_sep')
    cls_2 = separable_conv(header_2, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_2_sep')

    reg_3 = tf.keras.layers.Conv2D(3 * 4, kernel_size=3, padding='SAME',
                                   name='reg_3_convbias')(header_3)
    cls_3 = tf.keras.layers.Conv2D(3 * num_classes, kernel_size=3, padding='SAME',
                                   name='cls_3_convbias')(header_3)
    
    reg_list = [tf.keras.layers.Reshape([-1, 4])(reg) for reg in [reg_0, reg_1, reg_2, reg_3]]
    cls_list = [tf.keras.layers.Reshape([-1, num_classes])(cls) for cls in [cls_0, cls_1, cls_2, cls_3]]

    reg = tf.keras.layers.Concatenate(axis=1, name='face_boxes')(reg_list)
    cls = tf.keras.layers.Concatenate(axis=1)(cls_list)

    cls = tf.keras.layers.Softmax(axis=-1, name='face_scores')(cls)
    
    model = tf.keras.Model(inputs=[input_node], outputs=[reg, cls])

    model.summary()

    return model
