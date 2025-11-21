import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Conv1DTranspose,
    BatchNormalization,
    Input,
    Activation,
    Dropout,
    Concatenate,
    Layer,
    ZeroPadding1D,
    Cropping1D,
)
from tensorflow.keras.models import Model


class CropAndConcat(Layer):
    def call(self, inputs):
        net1, net2 = inputs
        
        # get shape
        net1_shape = tf.shape(net1)
        net2_shape = tf.shape(net2)
        
        # calculate triming offset and size
        offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, 0]
        size = [-1, net1_shape[1], net2_shape[2]]
        
        # crop net2
        net2_crop = tf.slice(net2, offsets, size)
        
        # 連結
        return Concatenate(axis=-1)([net1, net2_crop])

def PhaseNet(input_shape=(3001, 3), n_class=3):
    inputs = Input(shape=input_shape)

    # Initial Convolution Layer
    x = Conv1D(8, 7, padding='same', dilation_rate=1, name='initial_conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, name='initial_bn')(x)
    x = Activation('relu')(x)

    # Down Branch Level 1
    down1 = Conv1D(8, 7, padding='same', use_bias=False, name='down1_conv1')(x)
    down1 = BatchNormalization(epsilon=1e-3, name='down1_bn1')(down1)
    down1 = Activation('relu')(down1)
    down1_skip = down1
    down1 = ZeroPadding1D(padding=(3, 3), name='down1_pad')(down1)
    down1 = Conv1D(8, 7, strides=4, padding='valid', use_bias=False, name='down1_conv2')(down1)
    down1 = BatchNormalization(epsilon=1e-3, name='down1_bn2')(down1)
    down1 = Activation('relu')(down1)

    # Down Branch Level 2
    down2 = Conv1D(16, 7, padding='same', use_bias=False, name='down2_conv1')(down1)
    down2 = BatchNormalization(epsilon=1e-3, name='down2_bn1')(down2)
    down2 = Activation('relu')(down2)
    down2_skip = down2
    down2 = ZeroPadding1D(padding=(2, 3), name='down2_pad')(down2)
    down2 = Conv1D(16, 7, strides=4, padding='valid', use_bias=False, name='down2_conv2')(down2)
    down2 = BatchNormalization(epsilon=1e-3, name='down2_bn2')(down2)
    down2 = Activation('relu')(down2)

    # Down Branch Level 3
    down3 = Conv1D(32, 7, padding='same', use_bias=False, name='down3_conv1')(down2)
    down3 = BatchNormalization(epsilon=1e-3, name='down3_bn1')(down3)
    down3 = Activation('relu')(down3)
    down3_skip = down3
    down3 = ZeroPadding1D(padding=(1, 3), name='down3_pad')(down3)
    down3 = Conv1D(32, 7, strides=4, padding='valid', use_bias=False, name='down3_conv2')(down3)
    down3 = BatchNormalization(epsilon=1e-3, name='down3_bn2')(down3)
    down3 = Activation('relu')(down3)

    # Down Branch Level 4
    down4 = Conv1D(64, 7, padding='same', use_bias=False, name='down4_conv1')(down3)
    down4 = BatchNormalization(epsilon=1e-3, name='down4_bn1')(down4)
    down4 = Activation('relu')(down4)
    down4_skip = down4
    down4 = ZeroPadding1D(padding=(2, 3), name='down4_pad')(down4)
    down4 = Conv1D(64, 7, strides=4, padding='valid', use_bias=False, name='down4_conv2')(down4)
    down4 = BatchNormalization(epsilon=1e-3, name='down4_bn2')(down4)
    down4 = Activation('relu')(down4)

    # Down Branch Level 5
    down5 = Conv1D(128, 7, padding='same', use_bias=False, name='down5_conv1')(down4)
    down5 = BatchNormalization(epsilon=1e-3, name='down5_bn1')(down5)
    down5 = Activation('relu')(down5)

    # Up Branch Level 4
    up4 = Conv1DTranspose(64, 7, strides=4, padding='valid', use_bias=False, name='up4_conv1')(down5)
    up4 = BatchNormalization(epsilon=1e-3, name='up4_bn1')(up4)
    up4 = Activation('relu')(up4)
    up4 = Cropping1D(cropping=(1, 2), name='up4_crop')(up4)
    up4 = CropAndConcat()([down4_skip, up4])
    up4 = Conv1D(64, 7, padding='same', use_bias=False, name='up4_conv2')(up4)
    up4 = BatchNormalization(epsilon=1e-3, name='up4_bn2')(up4)
    up4 = Activation('relu')(up4)

    # Up Branch Level 3
    up3 = Conv1DTranspose(32, 7, strides=4, padding='valid', use_bias=False, name='up3_conv1')(up4)
    up3 = BatchNormalization(epsilon=1e-3, name='up3_bn1')(up3)
    up3 = Activation('relu')(up3)
    up3 = Cropping1D(cropping=(1, 2), name='up3_crop')(up3)
    up3 = CropAndConcat()([down3_skip, up3])
    up3 = Conv1D(32, 7, padding='same', use_bias=False, name='up3_conv2')(up3)
    up3 = BatchNormalization(epsilon=1e-3, name='up3_bn2')(up3)
    up3 = Activation('relu')(up3)

    # Up Branch Level 2
    up2 = Conv1DTranspose(16, 7, strides=4, padding='valid', use_bias=False, name='up2_conv1')(up3)
    up2 = BatchNormalization(epsilon=1e-3, name='up2_bn1')(up2)
    up2 = Activation('relu')(up2)
    up2 = Cropping1D(cropping=(1, 2), name='up2_crop')(up2)
    up2 = CropAndConcat()([down2_skip, up2])
    up2 = Conv1D(16, 7, padding='same', use_bias=False, name='up2_conv2')(up2)
    up2 = BatchNormalization(epsilon=1e-3, name='up2_bn2')(up2)
    up2 = Activation('relu')(up2)

    # Up Branch Level 1
    up1 = Conv1DTranspose(8, 7, strides=4, padding='valid', use_bias=False, name='up1_conv1')(up2)
    up1 = BatchNormalization(epsilon=1e-3, name='up1_bn1')(up1)
    up1 = Activation('relu')(up1)
    up1 = Cropping1D(cropping=(1, 2), name='up1_crop')(up1)
    up1 = CropAndConcat()([down1_skip, up1])
    up1 = Conv1D(8, 7, padding='same', use_bias=False, name='up1_conv2')(up1)
    up1 = BatchNormalization(epsilon=1e-3, name='up1_bn2')(up1)
    up1 = Activation('relu')(up1)

    # Output Layer
    outputs = Conv1D(n_class, 1, padding='same', name='output_conv')(up1)
    outputs = Activation('softmax', name='output_softmax')(outputs)

    return Model(inputs, outputs)