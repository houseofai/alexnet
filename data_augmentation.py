
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def transform(image, label):
    image = tf.image.resize(image, [256,256])
    return image, label

def transform_test(image, label):
    image = tf.image.resize(image, [227, 227])
    return image, label

def crop(image, label):
    image = tf.image.random_crop(image, size=[227, 227, 3])
    return image, label

def fliph(image, label):
    image = tf.image.flip_left_right(image)
    return image, label

def processing(ds_name, batch_size, crop_amount):

    log.info("* Loading dataset")
    ds_train = tfds.load(ds_name, split='test[:80%]', as_supervised=True)
    ds_test = tfds.load(ds_name, split='test[80%:]', as_supervised=True)

    log.info("* Transforming the dataset:")
    log.info("** Resize")
    ds_train = ds_train.map(transform)
    ds_test = ds_test.map(transform_test)

    log.info("** Flipping horizontally")
    ds_train = ds_train.concatenate(ds_train.map(fliph))

    log.info("** Cache")
    ds_train = ds_train.cache().shuffle(batch_size*2)

    # Memory intensive
    log.info("** Repeat {} times".format(crop_amount))
    ds_train = ds_train.repeat(crop_amount)

    log.info("** Cropping")
    ds_train = ds_train.map(crop)

    ds_file_size = tf.data.experimental.cardinality(ds_train)#*crop_amount*2
    log.info("* Train Dataset size estimation: {}".format(ds_file_size))
    ds_file_size = tf.data.experimental.cardinality(ds_test)#*crop_amount*2
    log.info("* Test Dataset size estimation: {}".format(ds_file_size))

    return ds_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), \
        ds_test.batch(1)
