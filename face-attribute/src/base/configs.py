from pathlib import Path, PurePath

import yaml

from .logger import Logger


default_conf_yaml = Path(PurePath(__file__).parent.parent.parent, "configs", "configs.yaml").resolve()
_conf = {}
yaml_loaded = False


def _load_yaml_config(config_location: str = default_conf_yaml):
    global yaml_loaded
    global _conf
    if config_location is not Path:
        config_location = Path(config_location).resolve()
    if not yaml_loaded:
        if _conf_is_exists(config_location):
            if config_location.is_file():
                with open(config_location, encoding="utf-8", mode="r") as file:
                    _conf = yaml.full_load(file)
                    yaml_loaded = True
            else:
                Logger.error(f"Config file ({config_location}) is invalid yaml format.")
        else:
            Logger.warning(f"Config file ({config_location}) is not exists.")
    return _conf


def set_conf(name: str, value):
    """add key-value to config"""
    _conf[name] = value


def get_conf(name: str):
    """read from the centralised config"""
    _load_yaml_config()
    return _conf[name]


def print_conf():
    print(_conf)


def _conf_is_exists(config_location: str):
    return config_location.exists()
