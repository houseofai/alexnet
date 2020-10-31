import tensorflow_datasets as tfds
tfds.load('imagenet_a', split='test', as_supervised=True)
