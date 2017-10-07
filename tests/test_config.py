"""Test the kopt configuration setup
"""
import kopt
import os


def test_config_file_exists():
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kopt"))
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kopt/config.yaml"))


def test_load_config():
    assert kopt.config.save_dir() == \
        os.path.join(os.path.expanduser('~'), ".kopt/data/")
    assert kopt.config.db_host() == "localhost"
    assert kopt.config.db_port() == 1234
