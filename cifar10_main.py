from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import urllib
import tarfile
import os
import functools
import itertools
import cv2
import ran

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="cifar10_ran_model", help="model directory")
parser.add_argument("--epochs", type=int, default=50, help="training epochs")
parser.add_argument("--batch", type=int, default=64, help="batch size")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def cifar10_input_fn(data_dir, training, num_epochs=1, batch_size=1):

    def download(url, data_dir):

        if not os.path.exists(data_dir):

            os.makedirs(data_dir)

        filename = url.split("/")[-1]
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):

            filepath, _ = urllib.request.urlretrieve(url, filepath)

            tarfile.open(filepath).extractall(data_dir)

    def get_filenames(data_dir, training):

        download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", data_dir)

        data_dir = os.path.join(data_dir, "cifar-10-batches-bin")

        if training:

            return [os.path.join(data_dir, "data_batch_{}.bin".format(i)) for i in range(1, 6)]

        else:

            return [os.path.join(data_dir, "test_batch.bin")]

    def preprocess(image, training):

        if training:

            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, (32, 32, 3))
            image = tf.image.random_flip_left_right(image)

        image = tf.image.per_image_standardization(image)

        return image

    def parse(bytes, training):

        record = tf.decode_raw(bytes, tf.uint8)

        image = record[1:]
        image = tf.reshape(image, (3, 32, 32))
        image = tf.transpose(image, (1, 2, 0))
        image = tf.cast(image, tf.float32)
        image = preprocess(image, training)

        label = record[0]
        label = tf.cast(record[0], tf.int32)

        return {"image": image}, label

    filenames = get_filenames(data_dir, training)
    dataset = tf.data.FixedLengthRecordDataset(filenames, 32 * 32 * 3 + 1)
    dataset = dataset.shuffle(50000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(functools.partial(parse, training=training))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator().get_next()


def cifar10_model_fn(features, labels, mode, params, channels_first):

    inputs = features["image"]

    if channels_first:

        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    ran_model = ran.Model(
        filters=16,
        initial_conv_param=ran.Model.ConvParam(
            kernel_size=3,
            strides=1
        ),
        initial_pool_param=ran.Model.PoolParam(
            pool_size=1,
            strides=1
        ),
        bottleneck=True,
        version=2,
        block_params=[
            ran.Model.BlockParam(
                blocks=5,
                strides=1
            ),
            ran.Model.BlockParam(
                blocks=5,
                strides=2
            ),
            ran.Model.BlockParam(
                blocks=5,
                strides=2
            )
        ],
        attention_block_params=[
            ran.Model.AttentionBlockParam(
                blocks=1
            ),
            ran.Model.AttentionBlockParam(
                blocks=1
            ),
            ran.Model.AttentionBlockParam(
                blocks=1
            )
        ],
        logits_param=ran.Model.DenseParam(
            units=10
        ),
        channels_first=channels_first
    )

    logits, maps_list, masks_list = ran_model(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN
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
    }

    if not channels_first:

        maps_list = [tf.transpose(maps, [0, 3, 1, 2]) for maps in maps_list]
        masks_list = [tf.transpose(masks, [0, 3, 1, 2]) for masks in masks_list]

    maps_dict = {"maps{}".format(i): maps for i, maps in enumerate(maps_list)}
    masks_dict = {"masks{}".format(i): masks for i, masks in enumerate(masks_list)}

    predictions.update(features)
    predictions.update(maps_dict)
    predictions.update(masks_dict)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    loss += tf.add_n([tf.nn.l2_loss(variable)
                      for variable in tf.trainable_variables()]) * params["weight_decay"]

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

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=params["learning_rate_fn"](
                    global_step=tf.train.get_global_step()
                ),
                momentum=params["momentum"]
            )

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )


def main(unused_argv):

    cifar10_classifier = tf.estimator.Estimator(
        model_fn=functools.partial(
            cifar10_model_fn,
            channels_first=False
        ),
        model_dir=args.model,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu
                ),
                device_count={
                    "GPU": 1
                }
            )
        ),
        params={
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "learning_rate_fn": functools.partial(
                tf.train.exponential_decay,
                learning_rate=0.1,
                decay_steps=50000,
                decay_rate=0.1
            )
        }
    )

    if args.train:

        cifar10_classifier.train(
            input_fn=functools.partial(
                cifar10_input_fn,
                data_dir="data",
                training=True,
                num_epochs=args.epochs,
                batch_size=args.batch
            ),
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={
                        "probabilities": "softmax_tensor"
                    },
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = cifar10_classifier.evaluate(
            input_fn=functools.partial(
                cifar10_input_fn,
                data_dir="data",
                training=False
            )
        )

        print(eval_results)

    if args.predict:

        predict_results = cifar10_classifier.predict(
            input_fn=functools.partial(
                cifar10_input_fn,
                data_dir="data",
                training=False
            )
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            def scale(in_val, in_min, in_max, out_min, out_max):
                return out_min + (in_val - in_min) / (in_max - in_min) * (out_max - out_min)

            image = predict_result["image"]
            maps_list = [predict_result["maps{}".format(i)] for i in range(3)]
            masks_list = [predict_result["masks{}".format(i)] for i in range(3)]

            image = scale(image, image.min(), image.max(), 0., 255.).astype(np.uint8)

            cv2.imwrite("outputs/image{}.jpeg".format(i), image)

            for j, maps in enumerate(maps_list):

                for k, map in enumerate(maps):

                    map = scale(map, map.min(), map.max(), 0., 255.).astype(np.uint8)

                    cv2.imwrite("outputs/map{}_{}_{}.jpeg".format(i, j, k), map)

            for j, masks in enumerate(masks_list):

                for k, mask in enumerate(masks):

                    mask = scale(mask, mask.min(), mask.max(), 0., 255.).astype(np.uint8)

                    cv2.imwrite("outputs/mask{}_{}_{}.jpeg".format(i, j, k), mask)


if __name__ == "__main__":

    tf.app.run()
