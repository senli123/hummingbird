"""工具类封装
"""
import glob
import logging
import os
import sys
import traceback
from typing import Callable, Optional, Union
from unittest.util import strclass

try:
    from torch import multiprocessing as mp
except ImportError:
    import multiprocess as mp

from hummingbird.utils.logging import get_logger


def get_root_logger(log_file = None, log_level = logging.INFO) -> logging.Logger:

    logger = get_logger(
        name = 'mmdeploy', log_file = log_file, log_level = log_level
    )

    return logger


def get_file_path(prefix, candidates) -> str:
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''

