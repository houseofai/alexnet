
import tensorflow_datasets as tfds
import logging
import tensorflow as tf
import sys

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

tf.executing_eagerly()

#builder = tfds.builder('imagenet_a')
#info = builder.info

#print(info)

#rint(info.features.shape)
#print(info.features.dtype)
#print(info.features['image'].shape)
#print(info.features['image'].dtype)

def transform(image, label):
    image = tf.image.resize(image, [256,256])
    return image, label

def crop(image, label):
    image = tf.image.random_crop(image, size=[227, 227, 3])
    return image, label

ds = tfds.load('imagenet_a', split='test', as_supervised=True)

log.info("** Tranform")
ds = ds.map(transform)
log.info("** Cropping")
ds = ds.map(crop)
#ds = ds.concatenate(ds1)

ds_final = ds.map(crop)

for i in range(0, 7):
    ds_new_crop = ds.map(crop)
    ds_final = ds_final.concatenate(ds_new_crop)

ds_final = ds_final.batch(128).shuffle(60000)

log.info("** Counting images")
i = 0
for image, label in ds_final:
    i+=1
    if i%10000==0:
        log.info("Calculating images: {}".format(i))

log.info("Amount of images: {}".format(i))
