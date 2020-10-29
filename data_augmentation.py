
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

def crop(image, label):
    image = tf.image.random_crop(image, size=[227, 227, 3])
    return image, label

def fliph(image, label):
    image = tf.image.flip_left_right(image)
    return image, label

def img_shape(ds):
    image, label = next(iter(ds))
    return image.shape

def processing():

    log.info("--- Dataset ---")
    log.info("* Loading ImageNet dataset")
    ds = tfds.load('imagenet_a', split='test', as_supervised=True)

    log.info("* Transforming the dataset:")
    log.info("** Resize")
    ds = ds.map(transform)

    log.info("** Cropping")
    ds_final = ds.map(crop)

    augment = True
    if augment:
        for i in range(0, 1047):
            ds_final = ds_final.concatenate(ds.map(crop))

        log.info("** Flipping horizontally")
        ds_final = ds_final.concatenate(ds_final.map(fliph))

    count_images = False
    if count_images:
        log.info("** Counting images")
        i = 0
        for image, label in ds_final:
            i+=1
            if i%10000==0:
                log.info("Calculating images: {}".format(i))

        log.info("Amount of images: {}".format(i))


    return ds_final.batch(128)#.prefetch()
