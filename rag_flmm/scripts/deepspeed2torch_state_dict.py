import argparse

import torch
from xtuner.model.utils import guess_load_checkpoint
from torch.serialization import add_safe_globals

try:
    from mmengine.config import Config
    from mmengine.config.config import ConfigDict
except Exception:
    Config = None
    ConfigDict = None

if Config is not None and ConfigDict is not None:
    try:
        add_safe_globals([Config, ConfigDict])
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--deepspeed_path", default="", type=str)
    parser.add_argument("--torch_path", default="", type=str)

    args = parser.parse_args()
    state_dict = guess_load_checkpoint(args.deepspeed_path)
    torch.save(state_dict, args.torch_path)
