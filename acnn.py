from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import argparse
import itertools
import functools
import operator
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000, help="training steps")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--model", type=str, default="mnist_acnn_model", help="model directory")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def acnn_model_fn(features, labels, mode, size, data_format):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN

    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, REDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 1
    (-1, 28, 28, 1) -> (-1, 28, 28, 32)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = features["images"]

    if data_format == "channels_first":

        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = utils.chunk_images(
        inputs=inputs,
        size=size,
        data_format=data_format
    )

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )

    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 2
    (-1, 28, 28, 32) -> (-1, 28, 28, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )

    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 1
    (-1, 28, 28, 64) -> (-1, 14, 14, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=inputs,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 2
    (-1, 14, 14, 3) -> (-1, 7, 7, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=attentions,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 3
    (-1, 7, 7, 3) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = attentions.get_shape().as_list()

    attentions = tf.layers.flatten(attentions)

    attentions = tf.layers.dense(
        inputs=attentions,
        units=10
    )

    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 4
    (-1, 10) -> (-1, 7, 7, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.dense(
        inputs=attentions,
        units=functools.reduce(operator.mul, shape[1:])
    )

    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1] + shape[1:]
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 5
    (-1, 7, 7, 3) -> (-1, 14, 14, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 6
    (-1, 14, 14, 9) -> (-1, 28, 28, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer
    (-1, 28, 28, 64), (-1, 28, 28, 9) -> (-1, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = utils.flatten_images(inputs, data_format)

    attentions = utils.flatten_images(attentions, data_format)

    inputs = tf.matmul(
        a=inputs,
        b=attentions,
        transpose_a=False if data_format == "channels_first" else True,
        transpose_b=True if data_format == "channels_first" else False
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dense layer 3
    (-1, 64, 9) -> (-1, 1024)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.flatten(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer 4
    (-1, 1024) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )

    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1
        ),
        "probabilities": tf.nn.softmax(
            logits=logits,
            name="softmax_tensor"
        ),
        "attentions": utils.chunk_images(
            inputs=attentions,
            size=size,
            data_format="channels_last"
        )
    }

    predictions.update(features)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )

    if mode == tf.estimator.ModeKeys.TRAIN:

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            train_op = tf.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )


def main(unused_argv):

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = mnist.train.images
    eval_images = mnist.test.images
    train_labels = mnist.train.labels.astype(np.int32)
    eval_labels = mnist.test.labels.astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=functools.partial(
            acnn_model_fn,
            size=[28, 28],
            data_format="channels_last"
        ),
        model_dir=args.model,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu
                )
            )
        )
    )

    if args.train:

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": train_images},
            y=train_labels,
            batch_size=args.batch,
            num_epochs=None,
            shuffle=True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "probabilities": "softmax_tensor"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            max_steps=args.steps,
            hooks=[logging_hook]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": eval_images},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )

        eval_results = mnist_classifier.evaluate(
            input_fn=eval_input_fn
        )

        print(eval_results)

    if args.predict:

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": eval_images},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )

        predict_results = mnist_classifier.predict(
            input_fn=predict_input_fn
        )

        for predict_result in predict_results:

            def scale(in_val, in_min, in_max, out_min, out_max):
                return out_min + (in_val - in_min) / (in_max - in_min) * (out_max - out_min)

            image = predict_result["images"]
            attention = predict_result["attentions"]

            image = image.repeat(repeats=3, axis=-1)
            image[:, :, 0] += np.apply_along_axis(func1d=np.sum, axis=-1, arr=attention)

            cv2.imshow("image", image)

            if cv2.waitKey(1000) == ord("q"):

                break


if __name__ == "__main__":
    tf.app.run()
