
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TrainObj():

    def __init__(self, strategy, model, loss, optimizer, global_batch_size):

        self.strategy = strategy
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.global_batch_size = global_batch_size
