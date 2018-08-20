from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools


def batch_normalization(data_format):

    return functools.partial(
        tf.layers.batch_normalization,
        axis=1 if data_format == "channels_first" else 3
    )


def global_average_pooling2d(data_format):

    return functools.partial(
        tf.reduce_mean,
        axis=[2, 3] if data_format == "channels_first" else [1, 2]
    )


def up_sampling2d(size, data_format):

    return tf.keras.layers.UpSampling2D(
        size=size,
        data_format=data_format
    )
