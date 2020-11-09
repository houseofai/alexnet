
from munch import munchify
import yaml
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ConfigManager:
    """
    Class to load the configuration from a file
    """
    def __init__(self, conf_file):
        """
        Initialize the file to read the parameters from
        :param conf_file: The name of the file: 'original' or 'test'
        """
        log.info("--- Configuration file ---")
        config_file = "original"
        if conf_file.lower() == "test":
            config_file = "test"

        log.info("* Loading configuration file '{}'".format(config_file))
        self.config = munchify(yaml.safe_load(open("config/{}.yml".format(config_file))))
        log.info("** Loaded")

    def get_conf(self):
        """
        Get the configuration parameters
        :return: The configuration parameters
        """
        return self.config
