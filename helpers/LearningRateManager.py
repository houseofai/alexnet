import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LearningRateDecay:
    def __init__(self, patience):
        self.patience = patience
        self.wait = 0
        self.best_loss = np.Inf

    def early_stop(self, loss):
        early_stop = False
        if loss < self.best_loss:
            log.info("Current Loss ({}) better than previous loss ({})".format(loss, self.best_loss))
            self.best_loss = loss
            self.wait = 0
        elif self.wait >= self.patience:
            log.info("Loss hasn't decreased for the past {} epochs".format(self.patience))
            early_stop = True
        else:
            self.wait += 1

        return early_stop
