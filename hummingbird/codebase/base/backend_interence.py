"""针对不用backend的推断类封装
"""
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import mmcv
import torch