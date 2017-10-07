# -*- coding: utf-8 -*-

__author__ = 'Žiga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.1.0'

from . import utils
from . import model_data
from . import eval_metrics
from . import hyopt
from .hyopt import KMongoTrials, CompileFN, test_fn
from .config import db_host, set_db_host, db_port, set_db_port


# Setup logging
# TODO - do we need this?
import logging

log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('concise')
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)
