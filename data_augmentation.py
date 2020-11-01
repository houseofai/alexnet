
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

def processing(ds_name, batch_size, crop_amount):

    log.info("* Loading dataset")
    ds = tfds.load(ds_name, split='test', as_supervised=True)

    log.info("* Transforming the dataset:")
    log.info("** Resize")
    ds = ds.map(transform)

    log.info("** Flipping horizontally")
    ds = ds.concatenate(ds.map(fliph))

    #log.info("** Cache")
    #ds = ds.cache()

    log.info("** Repeat {} times".format(crop_amount))
    ds = ds.repeat(crop_amount)

    log.info("** Cropping")
    ds = ds.map(crop)

    #for i in range(0, crop_amount-1):
    #    ds_final = ds_final.concatenate(ds.map(crop))


    count_images = False
    if count_images:
        log.info("** Counting images")
        i = 0
        for image, label in ds_final:
            i+=1
            if i%10000==0:
                log.info("Calculating images: {}".format(i))

        log.info("Amount of images: {}".format(i))

    ds_file_size = tf.data.experimental.cardinality(ds)#*crop_amount*2
    log.info("* Dataset size estimation: {}".format(ds_file_size))

    return ds.batch(batch_size)#.prefetch(tf.data.experimental.AUTOTUNE) \
        #.shuffle(batch_size*2) \
