"""Configuration management for Kopt

Following the Keras configuration management:
https://github.com/fchollet/keras/blob/6f3e6bb6fc97e706f37dc078ae821f490b78035b/keras/backend/__init__.py#L14-L43
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import yaml
# --------------------------------------------


_kopt_base_dir = os.path.expanduser('~')
if not os.access(_kopt_base_dir, os.W_OK):
    _kopt_base_dir = '/tmp'

_kopt_dir = os.path.join(_kopt_base_dir, '.kopt')

# default model_sources
_DB_HOST = 'localhost'
_DB_PORT = 1234
_SAVE_DIR = os.path.join(_kopt_dir, "data/")


def db_host():
    return _DB_HOST


def set_db_host(_db_host):
    global _DB_HOST

    _DB_HOST = _db_host


def db_port():
    return _DB_PORT


def set_db_port(_db_port):
    global _DB_PORT

    _DB_PORT = _db_port


def save_dir():
    return _SAVE_DIR


def set_save_dir(_save_dir):
    global _SAVE_DIR

    _SAVE_DIR = _save_dir


# Attempt to read Kopt config file.
_config_path = os.path.expanduser(os.path.join(_kopt_dir, 'config.yaml'))
if os.path.exists(_config_path):
    try:
        _config = yaml.load(open(_config_path))
    except ValueError:
        _config = {}
    _db_host = _config.get('db_host', db_host())
    _db_port = _config.get('db_ip', db_port())
    assert isinstance(_db_port, int)
    set_db_host(_db_host)
    set_db_port(_db_port)


# Save config file, if possible.
if not os.path.exists(_kopt_dir):
    try:
        os.makedirs(_kopt_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

# Writing the file
if not os.path.exists(_config_path):
    _config = {
        'db_host': db_host(),
        "db_port": db_port(),
        "save_dir": save_dir()
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(yaml.dump(_config, indent=4, default_flow_style=False))
    except IOError:
        # Except permission denied.
        pass
