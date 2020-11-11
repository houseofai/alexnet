# Standard library imports
import logging
import argparse
import datetime

# Related third party imports
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Local application/library specific imports
from helpers import LogEvents as lev
from helpers import TimeManager as tm
from helpers import CheckpointManager as cm
from helpers import EarlyStopManager as esm
from config import ConfigManager as cfg
from data import data_augmentation as da
import train as tr


def debug(dir):
    """
    Set debug mode
    :param dir: Directory where debug files will be stored
    """
    mytime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.debugging.experimental.enable_dump_debug_info(dir + "/" + mytime, tensor_debug_mode="FULL_HEALTH",
                                                     circular_buffer_size=-1)


def predict(config, image_path):
    """
    Predict classes probabilities for one image
    :param config: Configuration parameters for the trainer and the checkpoint manager
    :param image_path: Path to the image
    """
    log.info("--- Load Model ---")
    training = tr.Train(config)
    cm.CheckpointManager(training.model, training.optimizer, config)

    log.info("--- Load and Transform the image ---")
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(227, 227)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    img = tf.keras.utils.get_file(origin=image_path)
    img = da.prepare_singleimage(img)

    log.info("--- Predict ---")
    predictions = training.predict(img)

    log.info("* Outputs Shape:", predictions.shape)
    log.info("* Outputs Max:", max(predictions[0]))
    log.info("* Outputs Index:", np.argmax(predictions[0]))


def train(config):
    """
    Train the AlexNet model
    :param config: Configuration parameters
    """
    training = tr.Train(config)

    log.info("--- Dataset ---")
    ds_train = da.prepare_trainset(config.data.dataset, config.data.split, config.training.batch_size,
                                   config.data.augment,
                                   config.data.crop_amount)
    ds_test = da.prepare_testset(config.data.dataset, config.data.split, config.training.batch_size,
                                 config.data.augment)
    ds_size = tf.data.experimental.cardinality(ds_train).numpy()

    # Helpers
    lrd = esm.EarlyStop(config.training.patience)
    ckpt_manager = cm.CheckpointManager(training.model, training.optimizer, config)
    logevents = lev.LogEvents(log_dir=config.tensorboard.dir)
    timemanager = tm.TimeManager(config.training.epochs, ds_size)

    # Train
    log.info("Start training")
    log.info("* epochs: {}".format(config.training.epochs))
    log.info("* Processing the images. Might take a while depending on the CPU")

    last_epoch = int(ckpt_manager.get_last_epoch())
    log.info("Training from epochs {} to {}".format(last_epoch, config.training.epochs))
    for epoch in range(last_epoch, config.training.epochs):
        log.info("Start of epoch {}".format(epoch))

        train_step = 0
        bar = tqdm(ds_train)
        for x_batch, y_batch in bar:
            training.train(x_batch, y_batch)
            bar.set_description("[epoch{}] Loss: {:.4} Acc.:{:.4}"
                                .format(epoch, training.train_loss.result(), training.train_accuracy.result()))

            # Display timing info
            # timemanager.display_batch(train_step, training.train_loss.result())

            # Save tensorboard event log
            logevents.log_train(train_step, training.train_loss.result(), training.train_accuracy.result())
            train_step += 1

        test_step = 0
        for x_batch, y_batch in tqdm(ds_test):
            training.test(x_batch, y_batch)

            # Save tensorboard event log
            logevents.log_test(test_step, training.test_loss.result(), training.test_accuracy.result())
            test_step += 1

        # Save checkpoint
        ckpt_manager.save(training.train_loss.result())
        # Display timing info
        timemanager.display(epoch, training.train_loss.result(), training.train_accuracy.result(),
                            training.test_loss.result(), training.test_accuracy.result())
        # Reset metrics
        logevents.reset(training.train_loss, training.train_accuracy, training.test_loss,
                        training.test_accuracy)

        if lrd.check(training.train_loss.result()):
            log.info("Early Stop !")
            break


if __name__ == '__main__':
    """
    Start of the program to train/predict/evaluate
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="orignal", help="'orignal' or 'test' config file")
    parser.add_argument("--mode", default="train", help="train or predict")
    parser.add_argument("--image", default="", help="path to an image (test mode)")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    config_manager = cfg.ConfigManager(args.conf)
    config = config_manager.get_conf()

    if config.tensorboard.debug:
        log.info("--- Debug activated ---")
        debug(config.tensorboard.dir)

    if args.mode == "predict":
        if args.image is not None:
            predict(config, args.image)
        else:
            ValueError("'image' argument needs to be specified")
    elif args.mode == "train":
        # Default mode
        train(config)
