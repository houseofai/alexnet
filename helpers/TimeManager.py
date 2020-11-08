# Standard library imports
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TimeManager:
    def __init__(self, total_epoch, dataset_size):
        self.last_ep_time = self.last_ba_time = time.perf_counter()
        self.total_epoch = total_epoch
        self.dataset_size = dataset_size

        if self.dataset_size > 100:
            self.log_step = self.dataset_size//100
        else:
            self.log_step = self.dataset_size

    def display(self, current_epoch):
        exec_time = time.perf_counter() - self.last_ep_time
        remain_time = (self.total_epoch-current_epoch)*exec_time
        log.info("[epoch:{}] - Execution time: {:.4} - ETA: {:.4}".format(current_epoch, exec_time, remain_time))
        self.last_ep_time = time.perf_counter()

    def display_batch(self, step, loss):
        if step % self.log_step == 0:
            exec_time = time.perf_counter() - self.last_ba_time
            remain_time = (self.dataset_size-step)*exec_time
            log.info("\t[Step:{}] - Execution time: {:.4} - ETA: {} - Loss: {:.4}".format(step, exec_time, remain_time, loss))
            self.last_ba_time = time.perf_counter()
