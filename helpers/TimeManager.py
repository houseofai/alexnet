import tensorflow as tf
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TimeManager():
    def __init__(self, total_epoch):
        self.start_time = time.perf_counter()
        self.last_ep_time = time.perf_counter()
        self.total_epoch = total_epoch

    def display(self, current_epoch):
        exec_time = time.perf_counter() - self.last_ep_time
        remain_time = (self.total_epoch-current_epoch)*exec_time
        log.info("[epoch:{}] - Execution time: {:.4} - ETA: {:.4}".format(current_epoch, exec_time, remain_time))
        self.last_ep_time = time.perf_counter()
