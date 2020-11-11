import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import resource

from data.PCAColorAugmentation import PCAColorAugmentation

"""
Methods to prepare and augment data
"""
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def resize(image, label):
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
        four_labels = np.repeat(labels, repeat)
        if all_patches is None:
            all_patches = four_patches
            all_labels = four_labels
        else:
            all_patches = np.concatenate((all_patches, four_patches), axis=0)
            all_labels = np.concatenate((all_labels, four_labels), axis=0)

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
    image = tf.image.resize(image, [227, 227])
    return image, label


def normalize(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def fliph(image, label):
    """
    Flip horizontally an image
    :param image: The image to flip
    :param label: A label (not affected)
    :return: The flipped image and the label
    """
    image = tf.image.flip_left_right(image)
    return image, label


pca = PCAColorAugmentation()


def pca_color_augmentation(image, label):
    """
    Augment colors on principal components of an image
    :param image: The image to find PCA on
    :param label: A label (not affected)
    :return: The augmented image and the label
    """
    image = pca.augmentation(image)
    return image, label


def pca_color_augmentation_numpy(image_array_input):
    """
    Principal Component Analysis
    Code taken from: https://github.com/koshian2/PCAColorAugmentation/blob/master/pca_aug_numpy_single.py
    :param image_array_input:
    :return: The image augmented
    """
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    assert image_array_input.dtype == np.uint8

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(U, rand * S)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out


def prepare_trainset(ds_name, split, batch_size, augment, crop_amount):
    """
    Prepare a dataset for training
    :param ds_name: The name of the dataset to download from Tensorflow Dataset
    :param batch_size: The batch size of the dataset
    :param augment: Boolean. Whether to augment data or not
    :param crop_amount: The amount of cropped images
    :return: The train set
    """
    # Fix for 'Too many open files' issue #1441: https://github.com/tensorflow/datasets/issues/1441
    #low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    #print("###LIMITS", low, high)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    log.info("* Loading train dataset")
    ds_train = tfds.load(ds_name, split='{}[:80%]'.format(split), as_supervised=True)

    log.info("* Transforming the dataset:")

    log.info("** Normalize")
    ds_test = ds_train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    log.info("** Resize")
    ds_train = ds_train.map(resize)

    if augment:
        log.info("** Augment colors on Principal Components")
        ds_train = ds_train.map(pca_color_augmentation)

        log.info("** Flipping horizontally")
        ds_train = ds_train.concatenate(ds_train.map(fliph))

    log.info("** Cache")
    ds_train = ds_train.cache().shuffle(batch_size * 2)

    if augment:
        # Memory intensive
        log.info("** Repeat {} times".format(crop_amount))
        ds_train = ds_train.repeat(crop_amount)

        log.info("** Cropping Train Set")
        ds_train = ds_train.map(crop)

    ds_file_size = tf.data.experimental.cardinality(ds_train)
    log.info("* Train Dataset size estimation: {}".format(ds_file_size))

    return ds_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def prepare_testset(ds_name, split, batch_size, augment):
    """
    Prepare a dataset for testing
    :param ds_name: The name of the dataset to download from Tensorflow Dataset
    :param batch_size: The batch size of the dataset
    :param augment: Boolean. Whether to augment data or not
    :return: The Test set
    """
    log.info("* Loading test dataset")
    ds_test = tfds.load(ds_name, split='{}[-20%:]'.format(split), as_supervised=True)

    log.info("* Transforming the dataset:")

    log.info("** Normalize")
    ds_test = ds_test.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    log.info("** Resize")
    ds_test = ds_test.map(resize)

    if augment:
        log.info("** Patching Test Set")
        ds_test_patches = patches(ds_test.batch(tf.cast(tf.divide(batch_size, 4), dtype=tf.int64)))

        log.info("** Cropping Test Set")
        ds_test = ds_test.map(center_crop)
        ds_test = ds_test.concatenate(ds_test_patches)

    ds_file_size = tf.data.experimental.cardinality(ds_test)
    log.info("* Test Dataset size estimation: {}".format(ds_file_size))
    return ds_test.batch(batch_size)


def prepare_singleimage(image):
    image, _ = normalize(image, None)
    image, _ = tf.image.resize(image, [227, 227])
    return image
