"""_summary_
    对ncnn的库进行加载
"""
import os
import shutil

def get_ops_path() -> str:
    """_summary_
        获取ncnn库的路径
    Returns:
        str: _description_
    """
    candidates = [
        '../../lib/libmmdeploy_ncnn_ops.so', '../../lib/mmdeploy_ncnn_ops.dll'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)


def get_onnx2ncnn_path() -> str:
    """_summary_
        获得onnx转ncnn工具的路径
    Returns:
        str: _description_
    """
    candidates = ['./mmdeploy_onnx2ncnn', './mmdeploy_onnx2ncnn.exe']
    onnx2ncnn_path = get_file_path(os.path.dirname(__file__), candidates)

    if onnx2ncnn_path is None or not os.path.exists(onnx2ncnn_path):
        onnx2ncnn_path = get_file_path('',candidates)
    
    if onnx2ncnn_path is None or not os.path.exists(onnx2ncnn_path):
        onnx2ncnn_path = shutil.which('mmdeploy_onnx2ncnn')
        onnx2ncnn_path = '' if onnx2ncnn_path is None else onnx2ncnn_path
    return onnx2ncnn_path


def get_ncnn2int8_path() -> str:
    ncnn2int8_path = shutil.which('ncnn2int8')
    if ncnn2int8_path is None:
        raise Exception(
            'Cannot find ncnn2int8, try `export PATH =/path/to/ncnn2int8`'
        )
    return ncnn2int8_path
