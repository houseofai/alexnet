
from munch import munchify
import yaml
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ConfigManager():
    def __init__(self, conf_file):
        log.info("--- Configuration file ---")
        config_file = "original"
        if conf_file.lower() == "test":
            config_file = "test"

        log.info("* Loading configuration file '{}'".format(config_file))
        self.config = munchify(yaml.safe_load(open("config/{}.yml".format(config_file))))
        log.info("** Loaded")

    def get_conf(self):
        return self.config
