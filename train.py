from model import AlexNet
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


epochs = 5
checkpoints_directory = './checkpoints'
checkpoint_name = 'alexnet'

# Load sample images
#china = load_sample_image("china.jpg")/255
#flower = load_sample_image("flower.jpg")/255

#china = np.expand_dims(china, axis=0)
#print(china.shape)
#images = np.array([china, flower])
#labels = np.array([0,1])
#print("Label Numpy: ", labels.shape)
# Technical conversion
#labels = tf.convert_to_tensor(labels)
#print("Label Tensor: ", labels.shape)
#batch_size, height, width, channels = images.shape
#print(china.shape)

def transform(image, label):
    image = tf.image.resize(image, [227,227])
    return image, label

def img_shape(ds):
    image, label = next(iter(ds))
    return image.shape

log.info("--- Dataset ---")
log.info("* Loading ImageNet dataset")
ds = tfds.load('imagenet_a', split='test', as_supervised=True, batch_size=128)

log.info("* Transforming the dataset")
ds = ds.map(transform)

#log.info("* Splitting the dataset")
#images = np.array([x.numpy() for x,y in ds])
#labels = np.array([y.numpy() for x,y in ds])
#ds = tf.expand_dims(ds, 0)
#patches = tf.image.extract_patches(images=images,
#                           sizes=[batch_size, 227, 227, channels],
#                           strides=[1, 1, 1, 1],
#                           rates=[1, 1, 1, 1],
#                           padding='VALID')


#print(images.shape)

#tf.keras.layers.Conv2D(48, 11, strides=4, activation="relu", input_shape=[224,224,3])
#filters= np.zeros(shape=(11,11,3, 48), dtype=np.float32)
#output = tf.nn.conv2d(china_resized, filters, strides=4, padding="VALID")
#print("#### ", output.shape)


log.info("--- Model ---")

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)
loss = tf.keras.losses.MeanSquaredError()

# Load model
log.info("* Building Alexnet model...")
model = AlexNet()
model.compile(optimizer=optimizer, loss=loss)
model.build(img_shape(ds))
log.info("* New model built")

log.info("* Summary")
log.info("{}".format(model.summary()))

# Checkpoint manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)#, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, directory=checkpoints_directory, checkpoint_name=checkpoint_name, max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    log.info("Restored from {}".format(manager.latest_checkpoint))
else:
    log.info("Initializing from scratch.")


class ManageCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.checkpoint_manager.save()

# Train
log.info("Start training")
log.info("* epochs: {}".format(epochs))
model.fit(ds, epochs=epochs, callbacks=[ManageCheckpoints(manager)])

# Save model
log.info("* Saving models")
#model.save_weights(ckpt_file)
manager.save()

# Predict
#outputs = model(images)
#print("Outputs:", outputs.shape)
# Plotting
#fig, ax = plt.subplots(1,2)
#ax[0].imshow(china)
#ax[1].imshow(outputs[0,:,:,1], cmap="gray")
#plt.show()
