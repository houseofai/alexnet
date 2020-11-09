import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import sys

"""
Methods to prepare and augment data
"""
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def transform(image, label):
    """
    Resize image to 256x256
    :param image: The image to resize
    :param label: A label (not affected)
    :return: The resized image and the label
    """
    image = tf.image.resize(image, [256, 256])
    return image, label


def crop(image, label):
    """
    Crop an image to 227x227
    :param image: The image to crop
    :param label: A label (not affected)
    :return: The cropped image and the label
    """
    image = tf.image.random_crop(image, size=[227, 227, 3])
    return image, label


def patches(ds):
    """
    Augment a dataset by creating four patches of size 227x227 (top-left, top right, bottom-left, bottom-right) from the images
    :param ds: The dataset containing the images and the labels
    :return: A new dataset with the patches and the corresponding labels
    """
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
    """
    Center crop an image
    :param image: The image to center crop
    :param label: A label (not affected)
    :return: The cropped image and the label
    """
    image = tf.image.central_crop(image, central_fraction=0.89)
    return image, label


def fliph(image, label):
    """
    Flip horizontally an image
    :param image: The image to flip
    :param label: A label (not affected)
    :return: The flipped image and the label
    """
    image = tf.image.flip_left_right(image)
    return image, label


def prepare_trainset(ds_name, batch_size, crop_amount):
    """
    Prepare a dataset for training
    :param ds_name: The name of the dataset to download from Tensorflow Dataset
    :param batch_size: The batch size of the dataset
    :param crop_amount: The amount of cropped images
    :return: The train set
    """
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
    """
    Prepare a dataset for testing
    :param ds_name: The name of the dataset to download from Tensorflow Dataset
    :param batch_size: The batch size of the dataset
    :return: The Test set
    """
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
