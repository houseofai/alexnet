

import yaml
from munch import munchify

config = munchify(yaml.safe_load(open("config/original.yml")))

print("Epochs: {}".format(config.training.checkpoint_name))
