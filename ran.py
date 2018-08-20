from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import util
import resnet


class Model(resnet.Model):

    """ implementation of RAN in TensorFlow

    [1] [Residual Attention Network for Image Classification](https://arxiv.org/pdf/1512.03385.pdf) 
        by Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang, Apr 2017.
    """

    AttentionModuleParam = collections.namedtuple(
        "AttentionModuleParam", (
            "initial_blocks",
            "medial_blocks",
            "attention_blocks",
            "final_blocks",
            "strides",
            "shortcuts"
        )
    )

    def __init__(self, initial_conv_param, initial_pool_param,
                 bottleneck, version, attention_module_params, final_block_param, logits_param, channels_first):

        self.initial_conv_param = initial_conv_param
        self.initial_pool_param = initial_pool_param
        self.bottleneck = bottleneck
        self.version = version
        self.attention_module_params = attention_module_params
        self.final_block_param = final_block_param
        self.logits_param = logits_param
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"

    def __call__(self, inputs, training):

        block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                    (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

        projection_shortcut = Model.projection_shortcut

        with tf.variable_scope("ran"):

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=self.initial_conv_param.filters,
                kernel_size=self.initial_conv_param.kernel_size,
                strides=self.initial_conv_param.strides,
                padding="same",
                data_format=self.data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
            )

            if self.version == 1:

                inputs = util.batch_normalization(self.data_format)(
                    inputs=inputs,
                    training=training,
                    fused=True
                )

                inputs = tf.nn.relu(inputs)

            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=self.initial_pool_param.pool_size,
                strides=self.initial_pool_param.strides,
                padding="same",
                data_format=self.data_format
            )

            maps_list = []
            masks_list = []

            for i, attention_module_param in enumerate(self.attention_module_params):

                filters = self.initial_conv_param.filters << i

                inputs, maps, masks = Model.attention_module(
                    inputs=inputs,
                    block_fn=block_fn,
                    initial_blocks=attention_module_param.initial_blocks,
                    medial_blocks=attention_module_param.medial_blocks,
                    attention_blocks=attention_module_param.attention_blocks,
                    final_blocks=attention_module_param.final_blocks,
                    filters=filters,
                    strides=attention_module_param.strides,
                    projection_shortcut=projection_shortcut,
                    shortcuts=attention_module_param.shortcuts,
                    data_format=self.data_format,
                    training=training
                )

                maps_list.append(maps)
                masks_list.append(masks)

            inputs = Model.block_layer(
                inputs=inputs,
                block_fn=block_fn,
                blocks=self.final_block_param.blocks,
                filters=filters << 1,
                strides=self.final_block_param.strides,
                projection_shortcut=projection_shortcut,
                data_format=self.data_format,
                training=training
            )

            if self.version == 2:

                inputs = util.batch_normalization(self.data_format)(
                    inputs=inputs,
                    training=training,
                    fused=True
                )

                inputs = tf.nn.relu(inputs)

            inputs = util.global_average_pooling2d(self.data_format)(inputs)

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.logits_param.units
            )

            return inputs, maps_list, masks_list

    @staticmethod
    def attention_block_layer(inputs, block_fn, blocks, filters, shortcuts, data_format, training):

        if shortcuts >= 2:

            shortcut2 = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=2,
            strides=2
        )

        for _ in range(blocks):

            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        if shortcuts >= 1:

            shortcut1 = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=2,
            strides=2
        )

        for _ in range(blocks):

            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        for _ in range(blocks):

            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        inputs = util.up_sampling2d(2, data_format)(inputs)

        if shortcuts >= 1:

            inputs += shortcut1

        for _ in range(blocks):

            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        inputs = util.up_sampling2d(2, data_format)(inputs)

        if shortcuts >= 2:

            inputs += shortcut2

        '''
        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            kernel_initializer=tf.variance_scaling_initializer(),
        )
        '''

        inputs = tf.nn.sigmoid(inputs)

        return inputs

    @staticmethod
    def attention_module(inputs, block_fn, initial_blocks, medial_blocks, attention_blocks, final_blocks,
                         filters, strides, projection_shortcut, shortcuts, data_format, training):

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=initial_blocks,
            filters=filters,
            strides=strides,
            projection_shortcut=projection_shortcut,
            data_format=data_format,
            training=training
        )

        maps = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=medial_blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        masks = Model.attention_block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=attention_blocks,
            filters=filters,
            shortcuts=shortcuts,
            data_format=data_format,
            training=training
        )

        inputs = tf.multiply(
            x=maps,
            y=masks
        )

        inputs += maps

        inputs = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=final_blocks,
            filters=filters,
            strides=1,
            projection_shortcut=None,
            data_format=data_format,
            training=training
        )

        return inputs, maps, masks
