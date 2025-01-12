import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from training.trainler import Trainler
from config.config_base import ConfigBase

for each in os.listdir("config"):
    exec(f"from config.{os.path.splitext(each)[0]} import *")

def main(config : ConfigBase):
    trainler = Trainler(this_config)
    trainler.learn()

def arg_parser() -> ConfigBase:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('flag', type=str, help='The `flag` of config ( config{number} )')
    args = parser.parse_args()
    this_flag = args.flag.replace("-","_")
    config_str  = f"this_config = config{this_flag}()"
    print(config_str)
    local_vars = {}
    exec(config_str, globals(), local_vars)
    this_config = local_vars['this_config']
    print(local_vars)
    print(this_config)
    return this_config

def test():
    trainler = Trainler(arg_parser())
    trainler.test_model()

if __name__ == "__main__":

    main(arg_parser())