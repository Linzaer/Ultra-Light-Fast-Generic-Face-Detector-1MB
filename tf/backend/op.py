import tensorflow as tf


def basic_conv(x, out_ch, kernel_size, stride=(1, 1), padding=0, dilation=1, relu=True,
               bn=True, prefix='basic_conv'):
    if 0 < padding:
        out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}_padding')(x)
    else:
        out = x
    out = tf.keras.layers.Conv2D(out_ch,
                                 kernel_size,
                                 strides=stride,
                                 dilation_rate=dilation,
                                 use_bias=(not bn),
                                 name=f'{prefix}_conv')(out)
    if bn:
        out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}_bn')(out)
    if relu:
        out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)

    return out


def basic_rfb(x, in_ch, out_ch, stride=1, scale=0.1, map_reduce=8, vision=1, prefix='basic_rfb'):
    inter_ch = in_ch // map_reduce

    branch0 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch0.0')
    branch0 = basic_conv(branch0, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch0.1')
    branch0 = basic_conv(branch0, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 1,
                         padding=vision + 1, relu=False, prefix=f'{prefix}.branch0.2')

    branch1 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch1.0')
    branch1 = basic_conv(branch1, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch1.1')
    branch1 = basic_conv(branch1, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 2,
                         padding=vision + 2, relu=False, prefix=f'{prefix}.branch1.2')

    branch2 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch2.0')
    branch2 = basic_conv(branch2, (inter_ch // 2) * 3, kernel_size=3, stride=1, padding=1,
                         prefix=f'{prefix}.branch2.1')
    branch2 = basic_conv(branch2, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch2.2')
    branch2 = basic_conv(branch2, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 4,
                         padding=vision + 4, relu=False, prefix=f'{prefix}.branch2.3')

    out = tf.keras.layers.Concatenate(axis=-1, name=f'{prefix}_cat')([branch0, branch1, branch2])
    out = basic_conv(out, out_ch, kernel_size=1, stride=1, relu=False, prefix=f'{prefix}.convlinear')
    shortcut = basic_conv(x, out_ch, kernel_size=1, stride=stride, relu=False, prefix=f'{prefix}.shortcut')
    out = tf.multiply(out, scale, name=f'{prefix}_mul')
    out = tf.keras.layers.Add(name=f'{prefix}_add')([out, shortcut])
    out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)

    return out


def separable_conv(x, out_ch, kernel_size, stride, padding, prefix='separable_conv'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}_dconv_padding')(x)

    out = tf.keras.layers.DepthwiseConv2D(kernel_size,
                                          strides=stride,
                                          name=f'{prefix}_dconvbias')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)
    out = tf.keras.layers.Conv2D(out_ch, 1,
                                 name=f'{prefix}_convbias')(out)

    return out


def conv_bn(x, out_ch, stride, padding=1, prefix='conv_bn'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}.0_padding')(x)
    out = tf.keras.layers.Conv2D(out_ch,
                                 (3, 3),
                                 strides=stride,
                                 use_bias=False,
                                 name=f'{prefix}.0_conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.1_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.2_relu')(out)

    return out


def conv_dw(x, out_ch, stride, padding=1, prefix='conv_dw'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}.0_padding')(x)
    out = tf.keras.layers.DepthwiseConv2D(3, strides=stride,
                                          use_bias=False,
                                          name=f'{prefix}.0_dconv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.1_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.2_relu')(out)

    out = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False, name=f'{prefix}.3_conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.4_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.5_relu')(out)

    return out
