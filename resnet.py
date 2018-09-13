from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import utils


class Model(object):

    """ implementation of ResNet in TensorFlow

    [1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) 
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    [2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) 
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """

    ConvParam = collections.namedtuple("ConvParam", ("kernel_size", "strides"))
    PoolParam = collections.namedtuple("PoolParam", ("pool_size", "strides"))
    BlockParam = collections.namedtuple("BlockParam", ("blocks", "strides"))
    DenseParam = collections.namedtuple("DenseParam", ("units"))

    def __init__(self, filters, initial_conv_param, initial_pool_param,
                 block_params, bottleneck, version,
                 logits_param, channels_first):

        self.filters = filters
        self.initial_conv_param = initial_conv_param
        self.initial_pool_param = initial_pool_param
        self.block_params = block_params
        self.bottleneck = bottleneck
        self.version = version
        self.logits_param = logits_param
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"

        self.block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                         (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

        self.projection_shortcut = Model.projection_shortcut

    def __call__(self, inputs, training):

        with tf.variable_scope("resnet"):

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=self.filters,
                kernel_size=self.initial_conv_param.kernel_size,
                strides=self.initial_conv_param.strides,
                padding="same",
                data_format=self.data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
            )

            if self.version == 1:

                inputs = utils.batch_normalization(self.data_format)(
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

            for i, block_param in enumerate(self.block_params):

                inputs = Model.block_layer(
                    inputs=inputs,
                    block_fn=self.block_fn,
                    blocks=block_param.blocks,
                    filters=self.filters << i,
                    strides=block_param.strides,
                    projection_shortcut=self.projection_shortcut,
                    data_format=self.data_format,
                    training=training
                )

            if self.version == 2:

                inputs = utils.batch_normalization(self.data_format)(
                    inputs=inputs,
                    training=training,
                    fused=True
                )

                inputs = tf.nn.relu(inputs)

            inputs = utils.global_average_pooling2d(self.data_format)(inputs)

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.logits_param.units
            )

            return inputs

    @staticmethod
    def building_block_v1(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters,
                strides=strides,
                data_format=data_format
            )

            shortcut = utils.batch_normalization(data_format)(
                inputs=shortcut,
                training=training,
                fused=True
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs += shortcut

        inputs = tf.nn.relu(inputs)

        return inputs

    @staticmethod
    def building_block_v2(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        return inputs

    @staticmethod
    def bottleneck_block_v1(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters << 2,
                strides=strides,
                data_format=data_format
            )

            shortcut = utils.batch_normalization(data_format)(
                inputs=shortcut,
                training=training,
                fused=True
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters << 2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs += shortcut

        inputs = tf.nn.relu(inputs)

        return inputs

    @staticmethod
    def bottleneck_block_v2(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters << 2,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = utils.batch_normalization(data_format)(
            inputs=inputs,
            training=training,
            fused=True
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters << 2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        return inputs

    @staticmethod
    def block_layer(inputs, block_fn, blocks, filters, strides, projection_shortcut, data_format, training):

        inputs = block_fn(
            inputs=inputs,
            filters=filters,
            strides=strides,
            projection_shortcut=projection_shortcut,
            data_format=data_format,
            training=training
        )

        for _ in range(1, blocks):

            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                strides=1,
                projection_shortcut=None,
                data_format=data_format,
                training=training
            )

        return inputs

    @staticmethod
    def projection_shortcut(inputs, filters, strides, data_format):

        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )
