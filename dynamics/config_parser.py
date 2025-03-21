import os
import pdb
from datetime import datetime
from functools import partial, reduce
from operator import getitem
from pathlib import Path

from utils.utils import *


class ConfigParser:
    def __init__(self, config, modification=None, resume=None, device=None, test=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.modification = modification
        self.resume = resume
        self.device = device
        self.test = test

    @classmethod
    def from_args(cls, parser, options=""):
        """
        Initialize this class from some cli arguments for perception.
        """
        for opt in options:
            parser.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(parser, tuple):
            args = parser.parse_args()

        # if args.device is not None:
        #     if args.device == "cpu":
        #         device = torch.device("cpu")
        #     else:
        #         device = torch.device("cuda:" + args.device)
        # else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg

        cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        print(f'Config read from {args.config}')

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }

        return cls(config, modification=modification, device=device)

    @classmethod
    def from_dynamics_args(cls, parser, options=""):
        """
        Initialize this class from some cli arguments for dynamics.
        """
        for opt in options:
            parser.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(parser, tuple):
            args = parser.parse_args()
            device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
            test_ckpt = args.test

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg

        cfg_fname = Path(args.config)

        config = read_json(cfg_fname)

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }
        return cls(config, modification=modification, device=device, test=test_ckpt)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def update_from_json(self, config_json):
        self._config.update(read_json(config_json))

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(".")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
