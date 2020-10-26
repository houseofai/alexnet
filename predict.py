from model import AlexNet
import tensorflow as tf
import logging
import sys

from sklearn.datasets import load_sample_image

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def transform(image, label):
    image = tf.image.resize(image, [227,227])
    return image, label

image = load_sample_image("flower.jpg")/255
image, label = transform(image, "")
image = tf.expand_dims(image, axis=0)


latest_ckpt = tf.train.latest_checkpoint("./checkpoints")
log.info("Loading the latest checkpoints: {}".format(latest_ckpt))
ckpt = tf.train.Checkpoint(step=tf.Variable(1))
ckpt.restore(latest_ckpt).expect_partial()

model = AlexNet()
outputs = model.predict(image)
print("Outputs Shape:", outputs.shape)
print("Outputs:", outputs)
# Plotting
#fig, ax = plt.subplots(1,2)
#ax[0].imshow(china)
#ax[1].imshow(outputs[0,:,:,1], cmap="gray")
#plt.show()
