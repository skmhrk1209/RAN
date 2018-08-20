from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import util
import resnet


class Model(resnet.Model):

    AttentionBlockParam = collections.namedtuple("AttentionBlockParam", ("blocks"))

    def __init__(self, initial_conv_param, initial_pool_param,
                 bottleneck, version, block_params, attention_block_params, logits_param, channels_first):

        self.initial_conv_param = initial_conv_param
        self.initial_pool_param = initial_pool_param
        self.bottleneck = bottleneck
        self.version = version
        self.block_params = block_params
        self.attention_block_params = attention_block_params
        self.logits_param = logits_param
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"

    def __call__(self, inputs, training):

        block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                    (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

        projection_shortcut = Model.projection_shortcut

        with tf.variable_scope("resnet"):

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

            for i, (block_param, attention_block_param) in enumerate(zip(self.block_params, self.attention_block_params)):

                filters = self.initial_conv_param.filters << i

                maps = Model.block_layer(
                    inputs=inputs,
                    block_fn=block_fn,
                    blocks=block_param.blocks,
                    filters=filters,
                    strides=block_param.strides,
                    projection_shortcut=projection_shortcut,
                    data_format=self.data_format,
                    training=training
                )

                masks = Model.attention_block_layer(
                    inputs=maps,
                    block_fn=block_fn,
                    blocks=attention_block_param.blocks,
                    filters=filters,
                    data_format=self.data_format,
                    training=training
                )

                inputs = (1 + masks) * maps

                maps_list.append(maps)
                masks_list.append(masks)

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
    def attention_block_layer(inputs, block_fn, blocks, filters, data_format, training):

        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=2,
            strides=2
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

        shortcut = Model.block_layer(
            inputs=inputs,
            block_fn=block_fn,
            blocks=1,
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

        inputs = util.up_sampling2d(2, data_format)(inputs)

        inputs += shortcut

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

        inputs = util.up_sampling2d(2, data_format)(inputs)

        inputs = tf.nn.sigmoid(inputs)

        return inputs
