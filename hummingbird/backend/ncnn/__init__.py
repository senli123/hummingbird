"""export 当前backend的接口和变量
"""
import importlib
import os.path as osp
from tkinter.messagebox import RETRY

from .init_plugins import get_onnx2ncnn_path, get_ops_path, get_torchscript2ncnn_path

from .onnx2ncnn import from_onnx
from .torchscript2ncnn import from_torchscrip

def is_available():
    """check当前的转换工具是否安好
    """

    has_pyncnn = importlib.util.find_spec('ncnn') is not None

    onnx2ncnn = get_onnx2ncnn_path()
    torchscript2ncnn = get_torchscript2ncnn_path()
    return has_pyncnn and osp.exists(onnx2ncnn) and osp.exists(torchscript2ncnn)

def is_custom_ops_available():
    """check扩展的ncnn是否安装
    """
    has_pyncnn_ext = importlib.util.find_spec(
        'hummingbird.backend.ncnn.ncnn_ext') is not None
    ncnn_ops_path = get_ops_path()
    return has_pyncnn_ext and osp.exists(ncnn_ops_path)

__all__ = ['from_onnx', 'from_torchscrip']

if is_available():
    try:
        from .wrapper import NCNNWrapper
        __all__ +=['NCNNWrapper']
    except Exception:
        pass


