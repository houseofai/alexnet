import tensorflow as tf
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CheckpointManager:
    """
    Class to manage the checkpoints
    """
    def __init__(self, model, optimizer, config):
        """
        Initialize the parameter for the checkpoint manager
        :param model: The model to train and save weights
        :param optimizer: The model optimizer
        :param config: The configuration parameters
        """
        log.info("--- Checkpoint ---")
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=config.checkpoint.dir,
                                                  checkpoint_name=config.checkpoint.name,
                                                  max_to_keep=config.checkpoint.max_to_keep)
        self.last_loss = np.Inf
        self.model = model
        self.ckpt_dir = config.checkpoint.dir
        self.ckpt_name = config.checkpoint.name

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            last_epoch = int(self.ckpt.step.numpy())
            log.info("* Restore checkpoint #{} at epoch {}".format(self.manager.latest_checkpoint, last_epoch))
        else:
            log.info("* No checkpoint found. Initializing from scratch.")

    def get_last_epoch(self):
        """
        Get the last save epoch
        :return: The last epoch number
        """
        last_epoch = 0
        if self.manager.latest_checkpoint:
            last_epoch = int(self.ckpt.step.numpy())
        return last_epoch

    def save(self, loss):
        """
        Save the current checkpoint
        :param loss: The loss to compare with the best checkpoint
        """
        self.ckpt.step.assign_add(1)
        self.__save_best(loss)
        save_path = self.manager.save()
        log.info("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def __save_best(self, loss):
        """
        Save the checkpoint if the loss is better than the previous one
        :param loss: The loss to compare
        """
        if loss < self.last_loss:
            log.info("Saving best weights")
            self.model.save_weights("{}/best-weights/{}-{}".format(self.ckpt_dir, self.ckpt_name, int(self.ckpt.step)))
            self.last_loss = loss
