from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import resnet


class Model(resnet.Model):

    """ implementation of RAN in TensorFlow (improved version)

    [1] [Residual Attention Network for Image Classification](https://arxiv.org/pdf/1512.03385.pdf) 
        by Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, 
        Honggang Zhang, Xiaogang Wang, and Xiaoou Tang, Apr 2017.
    """

    AttentionBlockParam = collections.namedtuple("AttentionBlockParam", ("blocks"))

    def __init__(self, filters, initial_conv_param, initial_pool_param,
                 block_params, attention_block_params, bottleneck, version, logits_param):

        self.filters = filters
        self.initial_conv_param = initial_conv_param
        self.initial_pool_param = initial_pool_param
        self.block_params = block_params
        self.attention_block_params = attention_block_params
        self.bottleneck = bottleneck
        self.version = version
        self.logits_param = logits_param

        self.block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                         (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

        self.projection_shortcut = Model.projection_shortcut

    def __call__(self, inputs, data_format, training):

        with tf.variable_scope("ran"):

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=self.filters,
                kernel_size=self.initial_conv_param.kernel_size,
                strides=self.initial_conv_param.strides,
                padding="same",
                data_format=data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
            )

            if self.version == 1:

                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    axis=1 if data_format == "channels_first" else 3,
                    training=training,
                    fused=True
                )

                inputs = tf.nn.relu(inputs)

            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=self.initial_pool_param.pool_size,
                strides=self.initial_pool_param.strides,
                padding="same",
                data_format=data_format
            )

            maps_list = []
            masks_list = []

            for i, (block_param, attention_block_param) in enumerate(zip(self.block_params, self.attention_block_params)):

                maps = Model.block_layer(
                    inputs=inputs,
                    block_fn=self.block_fn,
                    blocks=block_param.blocks,
                    filters=self.filters << i,
                    strides=block_param.strides,
                    projection_shortcut=self.projection_shortcut,
                    data_format=data_format,
                    training=training
                )

                masks = Model.attention_block_layer(
                    inputs=maps,
                    block_fn=self.block_fn,
                    blocks=attention_block_param.blocks,
                    filters=self.filters << i,
                    data_format=data_format,
                    training=training
                )

                inputs = (1 + masks) * maps

                maps_list.append(maps)
                masks_list.append(masks)

            if self.version == 2:

                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    axis=1 if data_format == "channels_first" else 3,
                    training=training,
                    fused=True
                )

                inputs = tf.nn.relu(inputs)

            inputs = tf.reduce_mean(
                input_tensor=inputs,
                axis=[2, 3] if data_format == "channels_first" else [1, 2]
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.logits_param.units
            )

            return inputs, maps_list, masks_list

    @staticmethod
    def attention_block_layer(inputs, block_fn, blocks, filters, data_format, training):

        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=2,
            strides=2,
            padding="same",
            data_format=data_format
        )

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=2,
            strides=2,
            padding="same",
            data_format=data_format
        )

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        inputs = tf.keras.layers.UpSampling2D(
            size=2,
            data_format=data_format
        )(inputs)

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        inputs = tf.keras.layers.UpSampling2D(
            size=2,
            data_format=data_format
        )(inputs)

        inputs = tf.nn.sigmoid(inputs)

        return inputs
