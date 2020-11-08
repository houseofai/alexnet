import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def transform(image, label):
    image = tf.image.resize(image, [256, 256])
    return image, label


def transform(image):
    image = tf.image.resize(image, [227, 227])
    return image


def crop(image, label):
    image = tf.image.random_crop(image, size=[227, 227, 3])
    return image, label


def patches(ds):
    all_patches = None
    all_labels = None
    repeat = 4
    for step, (images, labels) in enumerate(ds):
        four_patches = tf.image.extract_patches(images,
                                                sizes=[1, 227, 227, 1],
                                                strides=[1, 29, 29, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')
        if all_patches is None:
            all_patches = four_patches
            all_labels = np.repeat(labels, repeat)
        else:
            all_patches = np.concatenate((all_patches, four_patches), axis=0)
            all_labels = np.concatenate((all_labels, np.repeat(labels, repeat)), axis=0)

    all_patches = tf.reshape(all_patches, [all_patches.shape[0] * 2 * 2, 227, 227, 3])
    return tf.data.Dataset.from_tensor_slices((all_patches, all_labels))


def center_crop(image, label):
    image = tf.image.central_crop(image, central_fraction=0.89)
    return image, label


def fliph(image, label):
    image = tf.image.flip_left_right(image)
    return image, label


def prepare_trainset(ds_name, batch_size, crop_amount):
    log.info("* Loading train dataset")
    ds_train = tfds.load(ds_name, split='test[:80%]', as_supervised=True)

    log.info("* Transforming the dataset:")
    log.info("** Resize")
    ds_train = ds_train.map(transform)

    log.info("** Flipping horizontally")
    ds_train = ds_train.concatenate(ds_train.map(fliph))

    log.info("** Cache")
    ds_train = ds_train.cache().shuffle(batch_size * 2)

    # Memory intensive
    log.info("** Repeat {} times".format(crop_amount))
    ds_train = ds_train.repeat(crop_amount)

    log.info("** Cropping Train Set")
    ds_train = ds_train.map(crop)

    ds_file_size = tf.data.experimental.cardinality(ds_train)
    log.info("* Train Dataset size estimation: {}".format(ds_file_size))

    return ds_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def prepare_testset(ds_name, batch_size):
    log.info("* Loading test dataset")
    ds_test = tfds.load(ds_name, split='test[80%:]', as_supervised=True)

    log.info("* Transforming the dataset:")
    log.info("** Resize")
    ds_test = ds_test.map(transform)

    log.info("** Patching Test Set")
    ds_test_patches = patches(ds_test.batch(tf.cast(tf.divide(batch_size, 4), dtype=tf.int64)))

    log.info("** Cropping Test Set")
    ds_test = ds_test.map(center_crop)
    ds_test = ds_test.concatenate(ds_test_patches)

    ds_file_size = tf.data.experimental.cardinality(ds_test)
    log.info("* Test Dataset size estimation: {}".format(ds_file_size))
    return ds_test.batch(batch_size)
