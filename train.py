from model import AlexNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

ckpt_file = "models/alexnet_ckpt"
epochs = 5

# Load sample images
china = load_sample_image("china.jpg")/255
flower = load_sample_image("flower.jpg")/255

#china = np.expand_dims(china, axis=0)
#print(china.shape)
images = np.array([china, flower])
labels = np.array([0,1])
print("Label Numpy: ", labels.shape)
# Technical conversion
labels = tf.convert_to_tensor(labels)
print("Label Tensor: ", labels.shape)
#batch_size, height, width, channels = images.shape
#print(china.shape)

images = tf.image.resize(images, [227,227])
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

# Load model
log.info("Building Alexnet model...")
model = AlexNet()
model.compile(optimizer='SGD', loss='mean_squared_error')
model.build(images.shape)
log.info("* New model built")

log.info("* Summary")
log.info("{}".format(model.summary()))

# Load weights
log.info("* Checking for checkpoints...")
if Path(ckpt_file).exists():
    log.info("* Checkpoints found. Loading...")
    load_status = model.load_weights(ckpt_file)
    log.debug("* {}".format(load_status.assert_consumed()))


# Train
log.info("Start training")
log.info("* epochs: {}".format(epochs))
model.fit(images, labels, epochs=epochs)

# Save model
log.info("* Saving models")
model.save_weights(ckpt_file)

# Predict
outputs = model(images)


print("Outputs:", outputs.shape)
# Plotting
#fig, ax = plt.subplots(1,2)
#ax[0].imshow(china)
#ax[1].imshow(outputs[0,:,:,1], cmap="gray")
#plt.show()
