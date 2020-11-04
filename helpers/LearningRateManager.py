import traceback
import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, config, train_loss):
        super(CustomSchedule, self).__init__()

        self.lr = config.optimizer.learning_rate
        self.loss = train_loss

    def __call__(self, step):
        print("MY LOSS: ", self.loss.result())
        return self.lr
